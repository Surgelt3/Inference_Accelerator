#pragma once

#include "neural_net.h"
#include <chevan_utils_print.hpp>

#define OP_NOP  0x0
#define OP_MAC  0x1
#define OP_RELU 0x2

#define MAX_REG 16
class MemManager
{
  public:
  const uchar* base;
  float* constant0;
  float* constant1;
  void replace(void*data,size_t N)
  {

  }
  void schedule(void*data,size_t N)
  {

  }
  void*request(void*ref)
  {
    return 0;
  }
};
class Compiler
{
  MemManager manager;
  const void*writeBasePtr;
  const float *CONSTANT_0;
  const float *CONSTANT_1;

  float*tmpBuffer[10];

  public:
  Compiler(void*basePtr)
  {
    this->writeBasePtr = basePtr;

    // load permanent 0 and 1
  }

  void transferMem(const void *ptr, size_t N)
  {
    // start data transfer
    // add to mem addr translation book
  }

  void load(const void*ptr,int reg)
  {
    chprintln("LOAD ",ptr," r",reg);
  }
  void store(void*ptr,int reg)
  {

  }
  void mov(int regOut,int regIn)
  {

  }
  void MAC(void *start,size_t length,float bias)
  {
    manager.replace(start, length);
    manager.request(start);

    chprintln("MAC ",start," ",length," ",bias);
  }
  void APPLY()
  {

  }



  void MULI(int reg,float val,int regOut)
  {
    manager.schedule(&val,1);
    load(&val,(reg+1)%MAX_REG);
    // todo
    
  }

  void runCommandMAC(const NetCommand&comm)
  {
    for (Tensor *t : comm.referenceLayer->layer_input)
    {
      manager.schedule(t->data._start, ch_arrlength(float, t->data));
    }

    float *addrA = comm.mac.addrA;
    float* addrB = comm.mac.addrB;

    const int npuLimit=4;
    ch_array toWrite = ch_arrstack(long, npuLimit * 2);
    // ch_array toWrite = ch_arrstack(long, comm.mac.repeat * comm.mac.N * 2);
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
              valA = manager.constant0;
            else
              valA = (float*)manager.request(addrA) + comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize + vShift * comm.mac.vertShiftSize;
            ch_arrpush(long, toWrite, (uchar*)valA-manager.base);

            if (comm.mac.indexes[2 * i + 1] == -1)
                valB = manager.constant0;
            else if (comm.mac.indexes[2 * i + 1] == -2)
              valB = manager.constant1;
            else
              valB = (float*)manager.request(addrB) + comm.mac.indexes[2 * i + 1] + c * comm.mac.repeatShiftB;
            ch_arrpush(long, toWrite, (uchar *)valB - manager.base);

            if(ch_arrlength(long,toWrite)==npuLimit*2)
            {
              MAC(toWrite._start, ch_arrlength(long, toWrite), *comm.mac.addrC);
              toWrite._end=toWrite._start;
              // get pointer to result
              ch_arrpush(long, toWrite, 0);
              ch_arrpush(long, toWrite, (uchar *)manager.constant1 - manager.base);
            }
          }
        }
        for (int i = ch_arrlength(long, toWrite) / 2; i < npuLimit; i++)
        {
          ch_arrpush(long, toWrite, (uchar *)manager.constant0 - manager.base);
          ch_arrpush(long, toWrite, (uchar *)manager.constant0 - manager.base);
        }
        MAC(toWrite._start, ch_arrlength(long, toWrite), *comm.mac.addrC);
        // wait for completion
        // store result
        // *(comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut) = sum;
      }
    }

    // if (comm.type == GAP)
    // {
    //   MULI(9, 2 / comm.mac.N, regAInUse[0]);
    //   mov(9, regAInUse[0]);
    // }
    //   store(comm.mac.out + shift,9);
  }

  void runCommandCLIP(const NetCommand &comm)
  {
    //   addrA = comm.clip.addrA;
    //   if ((long)comm.clip.addrA < 10)
    //     addrA = tmpData[(long)comm.clip.addrA];

    //   for (int i = 0; i < comm.clip.N; i++)
    //   {
    //     float val = addrA[i];
    //     if (comm.clip.addrMin)
    //       val = MIN(val, *comm.clip.addrMin);
    //     if (comm.clip.addrMax)
    //       val = MAX(val, *comm.clip.addrMax);
    //     comm.clip.out[i] = val;
    //   }
    //   break;
    // case NetCommandType::ADD:
    //   addrA = comm.add.addrA;
    //   addrB = comm.add.addrB;
    //   if ((long)comm.add.addrA < 10)
    //     addrA = tmpData[(long)comm.add.addrA];
    //   if ((long)comm.add.addrB < 10)
    //     addrB = tmpData[(long)comm.add.addrB];
    //   for (int i = 0; i < comm.add.N; i++)
    //   {
    //     comm.add.out[i] = addrA[i] + addrB[i];
    //   }
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
      case NetCommandType::GAP:
        break;
      case NetCommandType::CLIP:
        runCommandCLIP(comm);
      //   addrA = comm.clip.addrA;
      //   if ((long)comm.clip.addrA < 10)
      //     addrA = tmpData[(long)comm.clip.addrA];

      //   for (int i = 0; i < comm.clip.N; i++)
      //   {
      //     float val = addrA[i];
      //     if (comm.clip.addrMin)
      //       val = MIN(val, *comm.clip.addrMin);
      //     if (comm.clip.addrMax)
      //       val = MAX(val, *comm.clip.addrMax);
      //     comm.clip.out[i] = val;
      //   }
      //   break;
      // case NetCommandType::ADD:
      //   addrA = comm.add.addrA;
      //   addrB = comm.add.addrB;
      //   if ((long)comm.add.addrA < 10)
      //     addrA = tmpData[(long)comm.add.addrA];
      //   if ((long)comm.add.addrB < 10)
      //     addrB = tmpData[(long)comm.add.addrB];
      //   for (int i = 0; i < comm.add.N; i++)
      //   {
      //     comm.add.out[i] = addrA[i] + addrB[i];
      //   }
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