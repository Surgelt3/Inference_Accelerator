#include "neural_net.h"
#include "chevan_utils_print.hpp"

void Net::calculate()
{
  float *tmpData[10] = {0};
  int commCount = 0;
  for (const NetCommand &comm : commands)
  {
    chprintln("Command: ", commCount);
    float sum = 0;
    float *addrA = 0;
    float *addrB = 0;
    switch (comm.type)
    {
    case MAC:
    case GAP:
      addrA = comm.mac.addrA;
      addrB = comm.mac.addrB;

      if ((long)comm.mac.addrA < 10)
        addrA = tmpData[(long)comm.mac.addrA];
      if ((long)comm.mac.addrB < 10)
        addrB = tmpData[(long)comm.mac.addrB];

      for (int shift = 0; shift < comm.mac.shifts + 1; shift++)
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
            valA = addrA[comm.mac.indexes[2 * i] + shift];
          if (comm.mac.indexes[2 * i + 1] == -1)
            valB = 0;
          else if (comm.mac.indexes[2 * i + 1] == -2)
            valB = 1;
          else
            valB = addrB[comm.mac.indexes[2 * i + 1] + shift];

          sum += valA * valB;
        }
        if (comm.mac.addrC)
          sum += *comm.mac.addrC;

        if (comm.type == GAP)
          sum /= comm.mac.N / 2;
        *(comm.mac.out + shift) = sum;
        sum = 0;
      }
      break;
    case CLIP:
      addrA = comm.clip.addrA;
      if ((long)comm.clip.addrA < 10)
        addrA = tmpData[(long)comm.clip.addrA];

      for (int i = 0; i < comm.clip.N; i++)
      {
        float val = addrA[i];
        if (comm.clip.addrMin)
          val = MIN(val, *comm.clip.addrMin);
        if (comm.clip.addrMax)
          val = MAX(val, *comm.clip.addrMax);
        comm.clip.out[i] = val;
      }
      break;
    case ADD:
      addrA = comm.add.addrA;
      addrB = comm.add.addrB;
      if ((long)comm.add.addrA < 10)
        addrA = tmpData[(long)comm.add.addrA];
      if ((long)comm.add.addrB < 10)
        addrB = tmpData[(long)comm.add.addrB];
      for (int i = 0; i < comm.add.N; i++)
      {
        comm.add.out[i] = addrA[i] + addrB[i];
      }
      break;
    case MOV:
      addrA = comm.mov.addrA;
      addrB = comm.mov.addrB;
      if ((long)comm.mov.addrA < 10)
      {
        tmpData[(long)comm.mov.addrA] = (float *)realloc(tmpData[(long)comm.mov.addrA], sizeof(float) * comm.mov.N);
        addrA = tmpData[(long)comm.mov.addrA];
      }
      if ((long)comm.mov.addrB < 10)
      {
        tmpData[(long)comm.mov.addrB] = (float *)realloc(tmpData[(long)comm.mov.addrB], sizeof(float) * comm.mov.N);
        addrB = tmpData[(long)comm.mov.addrB];
      }

      memcpy(addrA, addrB, sizeof(float) * comm.mov.N);
      break;
    case ADDI:
      addrA = comm.opImm.addrA;
      if ((long)comm.opImm.addrA < 10)
        addrA = tmpData[(long)comm.opImm.addrA];
      for (int i = 0; i < comm.opImm.N; i++)
      {
        comm.opImm.out[i] = comm.opImm.c + addrA[i];
      }

      break;
    case MULI:
      addrA = comm.opImm.addrA;
      if ((long)comm.opImm.addrA < 10)
        addrA = tmpData[(long)comm.opImm.addrA];
      for (int i = 0; i < comm.opImm.N; i++)
      {
        comm.opImm.out[i] = comm.opImm.c * addrA[i];
      }

      break;

    default:
      break;
    }
    commCount++;
  }
}