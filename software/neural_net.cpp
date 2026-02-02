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
              if (comm.mac.indexes[2 * i + 1] == -1)
                valB = 0;
              else if (comm.mac.indexes[2 * i + 1] == -2)
                valB = 1;
              else
                valB = addrB[comm.mac.indexes[2 * i + 1] + c * comm.mac.repeatShiftB];

              sum += valA * valB;
            }
          }
          if (comm.mac.addrC)
            sum += *comm.mac.addrC;
  
          if (comm.type == GAP)
            sum /= comm.mac.N / 2;
          *(comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut) = sum;
          sum = 0;
        }
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

    case GEMM:
    {
      addrA = comm.gemm.addrA; // (M,K)
      addrB = comm.gemm.addrB; // (K,N)
      float*addrC=comm.gemm.addrC;// (M,N)

      const int K = comm.gemm.transA ? comm.gemm.dimsA[0] : comm.gemm.dimsA[1];
      const int M = comm.gemm.transA ? comm.gemm.dimsA[1] : comm.gemm.dimsA[0];
      const int N = comm.gemm.transB ? comm.gemm.dimsB[0] : comm.gemm.dimsA[1];

      for (int X = 0; X < M; X++)
      {
        for (int Y = 0; Y < N; Y++)
        {
          float val = 0;
          for (int i = 0; i < K; i++)
          {
            int aX = comm.gemm.transA ? i : X;
            int aY = comm.gemm.transA ? Y : i;

            int bX = comm.gemm.transB ? X : i;
            int bY = comm.gemm.transB ? i : Y;

            val += comm.gemm.alpha * (addrA[aX + aY * comm.gemm.dimsA[0]]) * (addrB[bX + bY * comm.gemm.dimsB[0]]);
          }
          if (comm.gemm.dimsC[0] == M && comm.gemm.dimsC[1] == N)
            val += comm.gemm.beta * (addrC[X + Y * comm.gemm.dimsC[0]]);
          else if (comm.gemm.dimsC[0] == M && (comm.gemm.dimsC[1] == 1 || comm.gemm.dimsC[1] == 0))
            val += comm.gemm.beta * (addrC[X]);
          else if ((comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0) && comm.gemm.dimsC[1] == N)
            val += comm.gemm.beta * (addrC[Y]);
          else if ((comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0) && comm.gemm.dimsC[0] == comm.gemm.dimsC[1])
            val += comm.gemm.beta * (addrC[0]);
          else
            chprinterr("incompatible length for C tensor in gemm");
        }
      }
      break;
    }
    default:
      break;
    }
    commCount++;
  }
}