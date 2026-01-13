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
  void MAC(int *regA,int*regB,int regC)
  {
    chprint("MAC ");
    for(int i=0;i<4;i++)
      chprint("r",regA[i],", ");
    for (int i = 0; i < 4; i++)
      chprint("r",regB[i],", ");
    chprint("r",regC);
    chprintln();
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
    // float *addrA = comm.mac.addrA;
    // float* addrB = comm.mac.addrB;

    // for(const Tensor*t:comm.referenceLayer->layer_input)
    // {
    //   transferMem(t->data._start, ch_arrlength(float, t->data));
    // }

    // if ((long)comm.mac.addrA < 10)
    //   addrA = tmpBuffer[(long)comm.mac.addrA];
    // if ((long)comm.mac.addrB < 10)
    //   addrB = tmpBuffer[(long)comm.mac.addrB];

    // for (int shift = 0; shift < comm.mac.shifts + 1; shift++)
    // {
    //   int regAInUse[] = {1, 2, 3, 4};
    //   int regBInUse[] = {5, 6, 7, 8};
    //   int count = 0;
    //   int regOut = 9;
    //   for (int i = 0; i < comm.mac.N; i++)
    //   {
    //     if (comm.mac.indexes[2 * i] == -1)
    //       load(CONSTANT_0, regAInUse[count % (sizeof(regAInUse)/sizeof(int))]);
    //     else if (comm.mac.indexes[2 * i] == -2)
    //       load(CONSTANT_1, regAInUse[count % (sizeof(regAInUse)/sizeof(int))]);
    //     else
    //       load(addrA + comm.mac.indexes[2 * i] + shift, regAInUse[count % (sizeof(regAInUse)/sizeof(int))]);
    //     if (comm.mac.indexes[2 * i + 1] == -1)
    //       load(CONSTANT_0, regBInUse[count % (sizeof(regBInUse)/sizeof(int))]);
    //     else if (comm.mac.indexes[2 * i + 1] == -2)
    //       load(CONSTANT_1, regBInUse[count % (sizeof(regBInUse)/sizeof(int))]);
    //     else
    //       load(addrB + comm.mac.indexes[2 * i + 1] + shift, regBInUse[count % (sizeof(regBInUse)/sizeof(int))]);
    //     count++;

    //     if (count % (sizeof(regAInUse) / sizeof(int)) == 0)
    //     {
    //       MAC(regAInUse,regBInUse,regOut);
    //       mov(regOut,regAInUse[0]);
    //       load(CONSTANT_1,regBInUse[0]);
    //       count++;
    //     }
    //   }


    //   if (comm.mac.addrC)
    //   {
    //     load(comm.mac.addrC, regAInUse[count % (sizeof(regAInUse) / sizeof(int))]);
    //     load(CONSTANT_1, regBInUse[count % (sizeof(regBInUse) / sizeof(int))]);
    //     count++;
    //   }
    //   while (count % (sizeof(regAInUse) / sizeof(int)))
    //   {
    //     load(CONSTANT_0, regAInUse[count % (sizeof(regAInUse) / sizeof(int))]);
    //     load(CONSTANT_0, regBInUse[count % (sizeof(regBInUse) / sizeof(int))]);
    //     count++;
    //   }
    //   MAC(regAInUse, regBInUse, regOut);

    //   if (comm.type == GAP)
    //   {
    //     MULI(9, 2 / comm.mac.N, regAInUse[0]);
    //     mov(9, regAInUse[0]);
    //   }
    //   store(comm.mac.out + shift,9);
    // }
  }

  public:
  void compileModel(const Net&net)
  {
    for (const NetCommand &comm : net.commands)
    {
      switch (comm.type)
      {
      case NetCommandType::MAC:
      case NetCommandType::GAP:
        runCommandMAC(comm);
        break;
      case NetCommandType::CLIP:
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