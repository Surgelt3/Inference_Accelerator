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
  void *virt_addr;
  const uchar *base;
  const float *constant0;
  const float *constant1;
  uint BLOCK_SIZE;
  MemManager();
  MemManager(uchar* base);
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
  void MAC(void *start,size_t length,float bias)
  {
    int instrLoc = (uchar *)manager.use(start, length) - manager.base;
    chassert(instrLoc < (1 << 7), "instr loc int overflow");
    chassert(length < (1 << 7), "length int overflow");
    int bias_i = *((int*)&bias);
    uint32_t instr = 0;
    instr |= OP_MAC << 29;
    instr |= (instrLoc & 0x3F) << 22;
    instr |= (length & 0x3F) << 16;
    instr |= (bias_i & 0x3F) << 4;
    manager.writeInstr(instr);
  }

  void LOAD(void *dst, void *src)
  {
    int mem_dst = (uchar *)dst - manager.base;
    int mem_src = (uchar *)src - manager.base;
    uint32_t instr = 0;
    instr |= OP_LOAD << 29;
    instr |= (mem_dst & 0x3F) << 22;
    instr |= mem_src & (~(1 << 22));
    manager.writeInstr(instr);
  }

  void runCommandMAC(const NetCommand&comm)
  {
    size_t reservedSize = manager.BLOCK_SIZE - sizeof(uint32_t);
    for (Tensor *t : comm.referenceLayer->layer_input)
    {
      for (void *p = t->data._start; p < t->data._end; p = (uchar *)p + reservedSize)
      {
        manager.schedule(p, sizeof(float) * MIN(reservedSize, (uchar *)t->data._end - (uchar *)p));
      }
    }

    void *aEnd = comm.referenceLayer->layer_input[0]->data._end;
    void *bEnd = comm.referenceLayer->layer_input[1]->data._end;
    float *addrA = comm.mac.addrA;
    float* addrB = comm.mac.addrB;

    std::mutex semiphoreMutex;
    int semiphoreA[512] = {};
    int semiphoreB[4] = {};
    memset(semiphoreA, 0, 512 * sizeof(int));
    memset(semiphoreB, 0, 4 * sizeof(int));

    // hope to god we don't overflow
    ch_array toWrite = ch_arrstack(uint32_t, comm.mac.repeat * comm.mac.N * 2);
    uint outIndex = 0;
    for (int vShift = 0; vShift < comm.mac.vertShift; vShift++)
    {
      for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
      {
        for (int c = 0; c < comm.mac.repeat; c++)
        {
          for (int i = 0; i < comm.mac.N; i++)
          {
            const float *valA = 0;
            const float *valB = 0;
            if (comm.mac.indexes[2 * i] == -1)
              valA = manager.constant0;
            else if (comm.mac.indexes[2 * i] == -2)
              valA = manager.constant1;
            else
            {
              int index = comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize;
              size_t loc = (size_t)((float*)addrA + index) / reservedSize * reservedSize;
              valA = (float *)manager.use((void *)loc, MIN(reservedSize, (size_t)aEnd - loc));
              semiphoreMutex.lock();
              semiphoreA[index/reservedSize]++;
              semiphoreMutex.unlock();
              chassert(valA != NULL, "memory manager already freed array");
              valA += index % (reservedSize / sizeof(float));
            }
            ch_arrpush(uint32_t, toWrite, (uchar *)valA - manager.base);

            if (comm.mac.indexes[2 * i + 1] == -1)
                valB = manager.constant0;
            else if (comm.mac.indexes[2 * i + 1] == -2)
              valB = manager.constant1;
            else
            {
              int index=comm.mac.indexes[2 * i + 1] + c * comm.mac.repeatShiftB;
              size_t loc = (size_t)((float*)addrB + index) / reservedSize * reservedSize;
              valB = (float *)manager.use((void *)loc, MIN(reservedSize, (size_t)bEnd - loc));
              semiphoreMutex.lock();
              semiphoreB[index / reservedSize]++;
              semiphoreMutex.unlock();
              chassert(valB != NULL, "memory manager already freed array");
              valB += index % (reservedSize / sizeof(float));
            }
            ch_arrpush(uint32_t, toWrite, (uchar *)valB - manager.base);
          }
        }
        while (ch_arrlength(uint32_t, toWrite) % 8)
          ch_arrpush(uint32_t, toWrite, (uchar *)manager.constant0 - manager.base);
        MAC(toWrite._start, sizeof(uint32_t) * ch_arrlength(uint32_t, toWrite), *comm.mac.addrC);
        // WHAT IS THE LOCATION OF THE OUT
        LOAD(manager.outPtr + (++outIndex),NULL);
        if(outIndex>=DATA_OUT_LENGTH)
        {
          memcpy((comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut),manager.readOut(),sizeof(float)*DATA_OUT_LENGTH);
          manager.readComplete();
          outIndex = 0;
        }
        void*capturedWrite=toWrite._start;
        void*capturedEnd=toWrite._end;
        std::thread t = std::thread(
            [this, capturedWrite, capturedEnd, reservedSize, &semiphoreMutex, &semiphoreA, &semiphoreB, addrA, addrB]()
            {
              // will stall until MAC is complete
              manager.freeBuffer(capturedWrite);
              for (uint i=0;i<((size_t)capturedEnd-(size_t)capturedWrite)/sizeof(uint32_t);i++)
              {
                size_t index;
                if (i % 2)
                  index = (size_t)((uchar*)manager.get(addrA) - (((uint32_t *)capturedWrite)[i] + (size_t)manager.base));
                else
                  index = (size_t)((uchar*)manager.get(addrB) - (((uint32_t *)capturedWrite)[i] + (size_t)manager.base));
                // multithreading is pain peko
                semiphoreMutex.lock();
                if (i % 2)
                {
                  if (--semiphoreA[index / reservedSize] <= 0)
                  {
                    manager.release(addrA + (index % reservedSize));
                  }
                }
                else
                {
                  if (--semiphoreB[index / reservedSize] <= 0)
                  {
                    manager.release(addrB + (index % reservedSize));
                  }
                }
                semiphoreMutex.unlock();
              }

          free(capturedWrite); });
        t.detach();
        toWrite = ch_arrcreate(uint32_t, ch_arrlength(uint32_t, toWrite));
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