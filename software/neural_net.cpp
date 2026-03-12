#include "neural_net.h"
#include "chevan_utils_print.hpp"
#include "math.h"

void Net::useCommand(const NetCommand&comm)
{
  switch (comm.type)
  {
  case MAC:
  case GAP:
  {
    float *addrA = comm.mac.addrA;
    float *addrB = comm.mac.addrB;

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
        *(comm.mac.out + shift + vShift * comm.mac.vertShiftSizeOut) = sum;
      }
    }
    break;
  }
  case CLIP:
  {
    float*addrA = comm.clip.addrA;
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
    float*addrA = comm.add.addrA;
    float*addrB = comm.add.addrB;
    for (int i = 0; i < comm.add.N; i++)
    {
      comm.add.out[i] = addrA[i] + addrB[i];
    }
    break;
  }

  case GEMM:
  {
    float*addrA = comm.gemm.addrA;        // (K,M)
    float*addrB = comm.gemm.addrB;        // (N,K)
    float *addrC = comm.gemm.addrC; // (M,N)

    const int K = !comm.gemm.transA ? comm.gemm.dimsA[1] : comm.gemm.dimsA[0];
    const int M = !comm.gemm.transA ? comm.gemm.dimsA[0] : comm.gemm.dimsA[1];
    const int N = !comm.gemm.transB ? comm.gemm.dimsB[1] : comm.gemm.dimsB[0];

    for (int Y = 0; Y < N; Y++)
    {
      for (int X = 0; X < M; X++)
      {
        double val = 0;
        for (int i = 0; i < K; i++)
        {
          // I have no clue how the logic works out to be the right indices, but it works!
          float valA = (!comm.gemm.transA ? addrA[X + i * comm.gemm.dimsA[0]] : addrA[i + X * comm.gemm.dimsA[1]]);
          float valB = (!comm.gemm.transB ? addrB[Y + i * comm.gemm.dimsB[0]] : addrB[i + Y * comm.gemm.dimsB[1]]);

          val += valA * valB;
        }
        val *= comm.gemm.alpha;
        if (comm.gemm.dimsC[0] == M && comm.gemm.dimsC[1] == N)
          val += comm.gemm.beta * (addrC[X + Y * comm.gemm.dimsC[0]]);
        else if ((comm.gemm.dimsC[0] == M && (comm.gemm.dimsC[1] == 1 || comm.gemm.dimsC[1] == 0)) ||
                 (comm.gemm.dimsC[1] == M && (comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0)))
          val += comm.gemm.beta * (addrC[X]);
        else if (((comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0) && comm.gemm.dimsC[1] == N) ||
                 ((comm.gemm.dimsC[1] == 1 || comm.gemm.dimsC[1] == 0) && comm.gemm.dimsC[0] == N))
          val += comm.gemm.beta * (addrC[Y]);
        else if ((comm.gemm.dimsC[0] == 1 || comm.gemm.dimsC[0] == 0) && comm.gemm.dimsC[0] == comm.gemm.dimsC[1])
          val += comm.gemm.beta * (addrC[0]);
        else
          chprinterr("incompatible length for C tensor in gemm\n");

        comm.gemm.out[X + Y * M] = val;
      }
    }
    break;
  }
  case COPY:
  {
    float *addrA = comm.copy.addrA;
    float *addrB = comm.copy.addrB;
    memcpy(addrB, addrA, sizeof(float) * comm.copy.length);
    break;
  }
  default:
    break;
  }
}
void Net::calculate()
{
  int commCount = 0;
  for (const NetCommand &comm : commands)
  {
    useCommand(comm);
    // chprintln("Command ", comm.type, " : ", commCount);
    commCount++;
  }
}