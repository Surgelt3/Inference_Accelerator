#pragma once

#include "neural_net.h"
#include <chevan_utils_print.hpp>
#include <thread>
#include <mutex>

#define OP_MAC  0x0
#define OP_LOAD  0x1
#define OP_END 0x2

#define DATA_OUT_LENGTH 16
class MemManager
{
private:
  int fd=-1;
  ch_hash mappedAddresses;
public:
  float *outPtr;
  void *shared_addr;
  const uchar *base;
  const float *constant0;
  const float *constant1;
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
class Compiler
{
  MemManager manager;


  public:
  Compiler()
  {
    this->manager = MemManager();
  }
  void MAC(void *start,size_t length)
  {
    int instrLoc = (size_t)start;
    chassert(instrLoc < (1 << 7), "instr loc int overflow");
    chassert(length < (1 << 7), "length int overflow");
    uint32_t instr = 0;
    instr |= OP_MAC << 29;
    instr |= (instrLoc & 0x3F) << 22;
    instr |= (length & 0x3F) << 16;
    manager.writeInstr(instr);
  }

  void LOAD(void *dst, void *src)
  {
    int mem_dst = (uchar *)dst - (uchar*)manager.shared_addr;
    int mem_src = (uchar *)src - (uchar*)manager.shared_addr;
    uint32_t instr = 0;
    instr |= OP_LOAD << 29;
    instr |= (mem_dst & 0x3F) << 22;
    instr |= mem_src & (~(1 << 22));
    manager.writeInstr(instr);
  }

  void runCommandMAC(const NetCommand&comm)
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
        MAC((void*)((uchar*)manager.use(toWrite._start, sizeof(float) * ch_arrlength(float, toWrite))-(uchar*)manager.shared_addr), ch_arrlength(float, toWrite) - 1);
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

  public:
  void compileModel(const Net&net)
  {
    for (const NetCommand &comm : net.commands)
    {
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
      
      case NetCommandType::MOV:
      //   addrA = comm.mov.addrA;
      //   addrB = comm.mov.addrB;
      //   if ((long)comm.mov.addrA < 10)
      //   {
      //     tmpData[(long)comm.mov.addrA] = (float *)realloc(tmpData[(long)comm.mov.addrA], sizeof(float) * comm.mov.N);
      //     addrA = tmpData[(long)comm.mov.addrA];
      //   }
      //   if ((long)comm.mov.addrB < 10)
      //   {
      //     tmpData[(long)comm.mov.addrB] = (float *)realloc(tmpData[(long)comm.mov.addrB], sizeof(float) * comm.mov.N);
      //     addrB = tmpData[(long)comm.mov.addrB];
      //   }

      //   memcpy(addrA, addrB, sizeof(float) * comm.mov.N);
      //   break;
      // case NetCommandType::ADDI:
      //   addrA = comm.opImm.addrA;
      //   if ((long)comm.opImm.addrA < 10)
      //     addrA = tmpData[(long)comm.opImm.addrA];
      //   for (int i = 0; i < comm.opImm.N; i++)
      //   {
      //     comm.opImm.out[i] = comm.opImm.c + addrA[i];
      //   }

      //   break;
      // case NetCommandType::MULI:
      //   addrA = comm.opImm.addrA;
      //   if ((long)comm.opImm.addrA < 10)
      //     addrA = tmpData[(long)comm.opImm.addrA];
      //   for (int i = 0; i < comm.opImm.N; i++)
      //   {
      //     comm.opImm.out[i] = comm.opImm.c * addrA[i];
      //   }

        break;

      default:
        break;
      }
    }
  }
};