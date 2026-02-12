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
            sum /= comm.mac.N;
          sum = MIN(MAX(sum, 0), 6);
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
      // a is supposed to be M,K and b K,N but this works so who cares
      addrA = comm.gemm.addrA;        // (N,K)
      addrB = comm.gemm.addrB;        // (K,M)
      float *addrC = comm.gemm.addrC; // (M,N)

      const int K = comm.gemm.transA ? comm.gemm.dimsA[1] : comm.gemm.dimsA[0];
      const int N = comm.gemm.transA ? comm.gemm.dimsA[0] : comm.gemm.dimsA[1];
      const int M = comm.gemm.transB ? comm.gemm.dimsB[0] : comm.gemm.dimsB[1];
      
      for(int Y=0;Y<N;Y++)
      {
        for(int X=0;X<M;X++)
        {
          double val=0;
          for (int i=0;i<K;i++)
          {
            val += addrA[i + Y * comm.gemm.dimsA[0]] * addrB[X + i * comm.gemm.dimsB[0]];
          }
          val *= comm.gemm.alpha;
          val += comm.gemm.beta * addrC[X];
          comm.gemm.out[X + Y * 1000] = val;
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