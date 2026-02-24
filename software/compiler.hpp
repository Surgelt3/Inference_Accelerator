#pragma once

#include "neural_net.h"
#include <chevan_utils_print.hpp>
#include <thread>
#include <mutex>

#define OP_MAC  0b000
#define OP_START  0b001
#define OP_END 0b010

#define DATA_OUT_LENGTH 16
class MemManager
{
private:
  int fd=-1;
  ch_hash mappedAddresses;
public:
  size_t maxSize;
  float *outPtr;
  void *shared_addr;
  const uchar *base;
  uint BLOCK_SIZE;
  MemManager();
  MemManager(uchar* shared);
  ~MemManager();
  void writeInstr(uint32_t i);
  void schedule(void *data, size_t N);
  void lock(void*data);
  void *get(void *data);
  void *request(void *ref,size_t N);
  void*use(void*ref,size_t N);
  float *readOut();
  void readComplete();
  void freeLastAdded();
  void release(void *data);
  void freeBuffer(void*data);
  void freeAll();
};

#define onDevice 0
class Compiler
{
  MemManager manager;
  float oobArray[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

public:
  Compiler() : manager(MemManager((uchar *)malloc(0x5000)))
  {
    // this->manager = MemManager();
  }
  void MAC(float*startLoc0,float*startLoc1,float*startLoc2,size_t length,float*paramsLoc)
  {
    // int instrLoc = (size_t)start;
    // chassert(instrLoc < (1 << 7), "instr loc int overflow");
    // chassert(length < (1 << 7), "length int overflow");
    uint32_t instr = 0;
    instr |= OP_MAC << 29;
    // instr |= (instrLoc & 0x3F) << 22;
    // instr |= (length & 0x3F) << 16;
#if onDevice
    // manager.writeInstr(instr);
#else
    float sum=0;
    if(length==1)
    {
      sum += startLoc0[0] * paramsLoc[0]+paramsLoc[1];
      // ((int *)startLoc0)[-1] &= ~0x2;
    }
    else if(length==9)
    {
      sum+=paramsLoc[10];
      for (int i = 0; i < 3; i++)
      {
        sum += startLoc0[i] * paramsLoc[0 + i];
        sum += startLoc1[i] * paramsLoc[3 + i];
        sum += startLoc2[i] * paramsLoc[6 + i];
      }
      // ((int *)startLoc0)[-1] &= ~0x2;
      // ((int *)startLoc1)[-1] &= ~0x2;
      // ((int *)startLoc2)[-1] &= ~0x2;
    }
#endif
  }

  void LOAD(void *dst, void *src)
  {
    // int mem_dst = (uchar *)dst - (uchar*)manager.shared_addr;
    // int mem_src = (uchar *)src - (uchar*)manager.shared_addr;
    // uint32_t instr = 0;
    // instr |= OP_LOAD << 29;
    // instr |= (mem_dst & 0x3F) << 22;
    // instr |= mem_src & (~(1 << 22));
    // manager.writeInstr(instr);
  }

  // work on multiple channels
  // work on RELU
  void runCommandMAC(const NetCommand&comm)
  {
    float *addrA = comm.mac.addrA;
    float *addrB = comm.mac.addrB;
    ch_array toWriteParams = ch_arrcreate(float, 12);
    ch_arrget(float, toWriteParams, 10) = 0;
    ch_arrget(float, toWriteParams, 11) = 0;
    if (comm.mac.N == 1)
    {
      struct {
        float kernel;
        float bias;
        float _zeroPad[10];
      }params;
      memset(&params, 0, sizeof(params));
      params.kernel = addrB[comm.mac.indexes[1]];
      params.bias = comm.mac.addrC ? *comm.mac.addrC : 0;
      memcpy(toWriteParams._start, &params, sizeof(params));
    }
    else if (comm.mac.N == 9)
    {
      struct
      {
        float kernel[9];
        float bias;
        float _zeroPad[2];
      } params;
      memset(&params, 0, sizeof(params));
      for (int i = 0; i < 9; i++)
      {
        params.kernel[i] = addrB[comm.mac.indexes[1 + 2 * i]];
      }
      params.bias = comm.mac.addrC ? *comm.mac.addrC : 0;
      memcpy(toWriteParams._start, &params, sizeof(params));
    }
    else
    {
      chprinterr("unrecognized kernel size");
    }
    manager.schedule(oobArray, 9 * sizeof(float));
    manager.lock(oobArray);
    manager.schedule(toWriteParams._start, sizeof(float) * 12);
    manager.lock(toWriteParams._start);

    // const size_t managerMaxBlockSize = manager.maxSize - manager.BLOCK_SIZE * 3;
    const size_t managerMaxBlockSize = manager.BLOCK_SIZE - 8;
    float *addrALoaded = addrA;
    manager.schedule(addrALoaded, MIN(sizeof(float) * (comm.mac.horShifts + 1), managerMaxBlockSize));
    for (int c = 0; c < comm.mac.repeat; c++)
    {
      float *loadedArrays[3]={0,0,0};
      float *outAddr[3];
      float *outParams;
      bool outRepeats[3] = {1, 1, 1};

      // prepare and load the data
      if (comm.mac.N == 1)
      {
        ch_arrget(float, toWriteParams, 0) = addrB[comm.mac.indexes[1] + (c % comm.mac.repeatB) * comm.mac.repeatShiftB];
        outParams = (float *)manager.use(toWriteParams._start, sizeof(float) * 3);
        if (comm.mac.indexes[0] == -1)
        {
          outAddr[0] = (float *)manager.request(oobArray, 1 * sizeof(float));
          outRepeats[0] = 0;
        }
        else if (comm.mac.indexes[0] == -2)
        {
          chprinterr("this should never happen");
        }
        else
        {
          int index = comm.mac.indexes[0] + c * comm.mac.repeatShiftA;
          assert(comm.mac.horShiftSize == 1); // should always be true
          if (addrA + index < addrALoaded)
          {
            addrALoaded = addrA + index - index % (managerMaxBlockSize / sizeof(float));
            manager.schedule(addrALoaded, MIN(sizeof(float) * (comm.mac.horShifts + 1), managerMaxBlockSize));
          }
          // outAddr[0] = (float *)manager.use(addrALoaded, 0) + index;
          // outAddr[0] = (float *)manager.request(addrALoaded, addrA + index - addrALoaded) + (addrA + index - addrALoaded) / sizeof(float);
          outRepeats[0] = 1;
        }
        outAddr[1] = outAddr[0];
        outAddr[2] = outAddr[1];
        outRepeats[1] = 0;
        outRepeats[2] = 0;
      }
      else if (comm.mac.N == 9)
      {
        for (int i = 0; i < 9; i++)
        {
          ch_arrget(float, toWriteParams, i) = addrB[comm.mac.indexes[1 + 2 * i] + (c % comm.mac.repeatB) * comm.mac.repeatShiftB];
        }
        outParams = (float *)manager.use(toWriteParams._start, sizeof(float) * 11);
        for (int y = 0; y < 3; y++)
        {
          float tempArray[3];
          bool hasOOB = false;
          for (int x = 0; x < 3; x++)
          {
            const int i = x + 3 * y;
            if (comm.mac.indexes[2 * i] == -1)
            {
              tempArray[x] = 0;
              hasOOB = true;
            }
            else if (comm.mac.indexes[2 * i] == -2)
            {
              tempArray[x] = 1;
              hasOOB = true;
            }
            else
            {
              tempArray[x] = addrA[comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA];
            }
          }
          if (hasOOB)
          {
            outRepeats[y] = 0;
            outAddr[y] = (float *)manager.request(tempArray, 3 * sizeof(float));
          }
          else
          {
            const int i0 = comm.mac.indexes[2 * (0 + 3 * y)] + c * comm.mac.repeatShiftA;
            const int i1 = comm.mac.indexes[2 * (1 + 3 * y)] + c * comm.mac.repeatShiftA;
            const int i2 = comm.mac.indexes[2 * (2 + 3 * y)] + c * comm.mac.repeatShiftA;
            assert(i0 + 1 == i1 && i1 + 1 == i2);
            if (addrA + i0 < addrALoaded)
            {
              addrALoaded = addrA + i0 - i0 % (managerMaxBlockSize / sizeof(float));
              manager.schedule(addrALoaded, MIN(sizeof(float) * (comm.mac.horShifts + 1 + 2), managerMaxBlockSize));
            }
            outRepeats[y] = 1;
          }
        }
      }
      else
      {
        chprinterr("unrecognized kernel size");
      }
      // execute
      for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
      {
        // ensure memory is loaded
        if (comm.mac.N == 1 && outRepeats[0] == 1)
        {
          int index = comm.mac.indexes[0] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize;
          assert(comm.mac.horShiftSize == 1); // should always be true
          if (addrA + index < addrALoaded)
          {
            // chprintln("shifted ", shift);
            addrALoaded = addrA + index - index % (managerMaxBlockSize / sizeof(float));
            manager.schedule(addrALoaded, MIN(sizeof(float) * (comm.mac.horShifts + 1), managerMaxBlockSize));
          }
          loadedArrays[0] = addrALoaded;
          outAddr[0] = (float *)manager.use(addrALoaded, (addrA + index - addrALoaded) / sizeof(float)) +
                       (addrA + index - addrALoaded) / sizeof(float);
          outAddr[1] = outAddr[0];
          outAddr[2] = outAddr[1];
        }
        else if (comm.mac.N == 9)
        {
          for (int y = 0; y < 3; y++)
          {
            if (outRepeats[y])
            {
              const int i0 = comm.mac.indexes[2 * (0 + 3 * y)] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize;
              const int i2 = comm.mac.indexes[2 * (2 + 3 * y)] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize;
              if (addrA + i0 < addrALoaded)
              {
                addrALoaded = addrA + i0 - i0 % (managerMaxBlockSize / sizeof(float));
                manager.schedule(addrALoaded, MIN(sizeof(float) * (comm.mac.horShifts + 1 + 2), managerMaxBlockSize));
              }
              loadedArrays[y] = addrALoaded;
              outAddr[y] = (float *)manager.use(addrALoaded, addrA + i2 - addrALoaded) + (addrA + i0 - addrALoaded) / sizeof(float);
            }
          }
        }

        MAC(outAddr[0] + (outRepeats[0] ? shift * comm.mac.horShiftSize : 0),
            outAddr[1] + (outRepeats[1] ? shift * comm.mac.horShiftSize : 0),
            outAddr[2] + (outRepeats[2] ? shift * comm.mac.horShiftSize : 0),
            comm.mac.N, outParams);
      }

      if (loadedArrays[0])
        manager.release(loadedArrays[0]);
      if (loadedArrays[1])
        manager.release(loadedArrays[1]);
      if (loadedArrays[2])
        manager.release(loadedArrays[2]);

      manager.release(addrALoaded);
    }

#if 0
    for (int vShift = 0; vShift < comm.mac.vertShift; vShift++)
    {
      for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
      {
        for (int c = 0; c < comm.mac.repeat; c++)
        {
          float *outAddr[3];
          if (comm.mac.N == 1)
          {
            if (comm.mac.indexes[0] == -1)
            {
              outAddr[0] = (float *)manager.use(oobArray, 1 * sizeof(float));
            }
            else if (comm.mac.indexes[0] == -2)
            {
              chprinterr("this should never happen");
            }
            else
            {
              int index = comm.mac.indexes[0] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize;
              outAddr[0] = (float *)manager.use(addrA + index, 1*sizeof(float));
            }
            outAddr[1] = (float *)manager.use(oobArray, 1*sizeof(float));
            outAddr[2] = outAddr[1];
          }
          else if (comm.mac.N == 9)
          {
            for (int y = 0; y < 3; y++)
            {
              float tempArray[3];
              bool hasOOB = false;
              for (int x = 0; x < 3; x++)
              {
                const int i = x + 3 * y;
                if (comm.mac.indexes[2 * i] == -1)
                {
                  tempArray[x] = 0;
                  hasOOB = true;
                }
                else if (comm.mac.indexes[2 * i] == -2)
                {
                  tempArray[x] = 1;
                  hasOOB = true;
                }
                else
                  tempArray[x] = addrA[comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize];
              }
              if(hasOOB)
              {
                // outAddr[y] = (float *)manager.use(tempArray, 3*sizeof(float));
              }
              else
              {
                const int i0 = comm.mac.indexes[2 * (0 + 3 * y)] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize;
                const int i1 = comm.mac.indexes[2 * (1 + 3 * y)] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize;
                const int i2 = comm.mac.indexes[2 * (2 + 3 * y)] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize;
                assert(i0 + 1 == i1 && i1 + 1 == i2);
                // outAddr[y] = (float *)manager.use(addrA + i0, 3*sizeof(float));
              }
            }
          }
          else
          {
            chprinterr("unrecognized kernel size");
          }

          // MAC()
        }
        // if (comm.mac.addrC)
        //   sum += *comm.mac.addrC;

        // *(comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut) = sum;
      }
      manager.freeAll();
    }
#endif
    manager.release(toWriteParams._start);
    manager.freeBuffer(toWriteParams._start);
    ch_arrfree(toWriteParams);
  }

#if 0
  void runCommandMAC1(const NetCommand&comm)
  {
    float *addrA = comm.mac.addrA;
    float* addrB = comm.mac.addrB;

    ch_array toWrite = ch_arrstack(float, comm.mac.repeat *comm.mac.N * 2 + 1);
    uint outIndex = 0;
    for (int vShift = 0; vShift < comm.mac.vertShift; vShift++)
    {
      for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
      {
        for (int c = 0; c < comm.mac.repeat; c++)
        {
          for (int i = 0; i < comm.mac.N; i++)
          {
            float valA = 0;
            float valB = 0;
            if (comm.mac.indexes[2 * i] == -1)
              valA = 0;
            else if (comm.mac.indexes[2 * i] == -2)
              valA = 1;
            else
              valA = addrA[comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize];
            ch_arrpush(float, toWrite, valA);

            if (comm.mac.indexes[2 * i + 1] == -1)
                valB = 0;
            else if (comm.mac.indexes[2 * i + 1] == -2)
              valB = 1;
            else
              valB = addrB[comm.mac.indexes[2 * i + 1] + c * comm.mac.repeatShiftB];
            ch_arrpush(float, toWrite, valB);
          }
        }
        while (ch_arrlength(float, toWrite) % 8)
          ch_arrpush(float, toWrite, 0.0);
        
        ch_arrpush(float,toWrite,*comm.mac.addrC);

        float *outLoc = comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut;
        // MAC((void*)((uchar*)manager.use(toWrite._start, sizeof(float) * ch_arrlength(float, toWrite))-(uchar*)manager.shared_addr), ch_arrlength(float, toWrite) - 1);
        // // WHAT IS THE LOCATION OF THE OUT
        // LOAD(manager.outPtr + (outIndex++), NULL);
        // if (outIndex >= DATA_OUT_LENGTH)
        // {
        //   // this will be wrong, pls fix
        //   // also if loop ends before this, we miss values
        //   memcpy((comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut - DATA_OUT_LENGTH), manager.readOut(), sizeof(float) * DATA_OUT_LENGTH);
        //   manager.readComplete();
        //   outIndex = 0;
        // }
        void*capturedWrite=toWrite._start;
        toWrite = ch_arrcreate(float, ch_arrlength(float, toWrite));
        std::thread t = std::thread(
            [this, capturedWrite, outLoc, outIndex]()
            {
              // will stall until MAC is complete
              manager.freeBuffer(capturedWrite);
              free(capturedWrite);
              
              // ig
              memcpy(outLoc, manager.readOut(), sizeof(float) * DATA_OUT_LENGTH);
              manager.readComplete();

            });
        t.detach();
      }
    }
    ch_arrfree(toWrite);
  }
#endif
  public:
  void compileModel(const Net&net)
  {
    int count=0;
    for (const NetCommand &comm : net.commands)
    {
      chprintln("command: ",count++);
      switch (comm.type)
      {
      case NetCommandType::MAC:
        runCommandMAC(comm);
        break;
      case NetCommandType::CLIP:
        // do nothing
        break;
      case NetCommandType::GAP:
        break;
      
      default:
        break;
      }
      // if(count>=20)
      //   break;
    }
  }
};