#include "neural_net.h"
#include "chevan_utils_print.hpp"
#include "math.h"

void Net::calculate()
{
  int commCount = 0;
  for (const NetCommand &comm : commands)
  {
    // chprintln("Command ", comm.type, " : ", commCount);
    float *addrA = 0;
    float *addrB = 0;
    switch (comm.type)
    {
    case MAC:
    case GAP:
    {
      addrA = comm.mac.addrA;
      addrB = comm.mac.addrB;

      for (int vShift = 0; vShift < comm.mac.vertShift; vShift++)
      {
        for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
        {
          double sum = 0;
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
                valB = addrB[comm.mac.indexes[2 * i + 1] + (c % comm.mac.repeatB) * comm.mac.repeatShiftB];
              sum += valA * valB;
            }
          }
          if (comm.mac.addrC)
            sum += *comm.mac.addrC;
  
          if (comm.type == GAP)
            sum /= comm.mac.N / 2;
          *(comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut) = sum;
        }
      }
      break;
    }
    case CLIP:
    {
      addrA = comm.clip.addrA;
      for (int i = 0; i < comm.clip.N; i++)
      {
        float val = addrA[i];
        if (comm.clip.addrMin)
          val = MAX(val, *comm.clip.addrMin);
        if (comm.clip.addrMax)
          val = MIN(val, *comm.clip.addrMax);
        comm.clip.out[i] = val;
      }
      break;
    }
    case ADD:
    {
      addrA = comm.add.addrA;
      addrB = comm.add.addrB;
      for (int i = 0; i < comm.add.N; i++)
      {
        comm.add.out[i] = addrA[i] + addrB[i];
      }
      break;
    }

    case GEMM:
    {
      addrA = comm.gemm.addrA;        // (N,K)
      addrB = comm.gemm.addrB;        // (K,M)
      float *addrC = comm.gemm.addrC; // (M,N)

      const int K = comm.gemm.transA ? comm.gemm.dimsA[1] : comm.gemm.dimsA[0];
      const int N = comm.gemm.transA ? comm.gemm.dimsA[0] : comm.gemm.dimsA[1];
      const int M = comm.gemm.transB ? comm.gemm.dimsB[1] : comm.gemm.dimsB[0];
      for(int X=0;X<1000;X++)
      {
        for(int Y=0;Y<1;Y++)
        {
          float val=0;
          for (int i=0;i<1280;i++)
          {
            val += (addrA[i + Y * 1280]) * (addrB[X + i * 1000]);
          }
          val *= comm.gemm.alpha;
          val += comm.gemm.beta * addrC[X];
          comm.gemm.out[X + Y * 1000] = val;
        }
      }
      // for (int Y = 0; Y < N; Y++)
      // {
      //   for (int X = 0; X < M; X++)
      //   {
      //     float val = 0;
      //     for (int i = 0; i < K; i++)
      //     {
      //       int aX = comm.gemm.transA ? Y : i;
      //       int aY = comm.gemm.transA ? i : Y;

      //       int bX = comm.gemm.transB ? i : X;
      //       int bY = comm.gemm.transB ? X : i;

      //       val += (addrA[aX + aY * comm.gemm.dimsA[0]]) * (addrB[bX + bY * comm.gemm.dimsB[0]]);
      //     }
      //     val *= comm.gemm.alpha;
          
      //     if (comm.gemm.dimsC[0] == M && comm.gemm.dimsC[1] == N)
      //       val += comm.gemm.beta * (addrC[X + Y * comm.gemm.dimsC[0]]);
      //     else if (comm.gemm.dimsC[0] == M && (comm.gemm.dimsC[1] == 1 || comm.gemm.dimsC[1] == 0))
      //       val += comm.gemm.beta * (addrC[X]);
      //     else if ((comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0) && comm.gemm.dimsC[1] == N)
      //       val += comm.gemm.beta * (addrC[Y]);
      //     else if ((comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0) && comm.gemm.dimsC[0] == comm.gemm.dimsC[1])
      //       val += comm.gemm.beta * (addrC[0]);
      //     else
      //       chprinterr("incompatible length for C tensor in gemm\n");
      //     comm.gemm.out[X + Y * M] = val;
      //   }
      // }
      break;
    }
    default:
      break;
    }
    commCount++;
  }
}