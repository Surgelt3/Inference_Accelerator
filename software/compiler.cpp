#include "compiler.hpp"
#include "importer.hpp"

static uint32_t currentPC = 0;
#define writeData(ptr, size)                   \
  {                                            \
    ++currentPC;                               \
    manager.writeData((float *)(ptr), (size)); \
  }
void Compiler::writeInstructions(const Net &net)
{
  float *loadedKernel = 0;
  for (const NetCommand &comm : net.commands)
  {
    switch (comm.type)
    {
    case NetCommandType::MAC:
    {
      float *addrA = comm.mac.addrA;
      float *addrB = comm.mac.addrB;
      for (int c = 0; c < comm.mac.repeat; c++)
      {
        for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
        {
          float *kernel = addrB + (comm.mac.indexes[1] + (c % comm.mac.repeatB) * comm.mac.repeatShiftB);
          if (loadedKernel != kernel || c != 0)
          {
            if (comm.mac.N == 9)
            {
              manager.writeInstruction(LOAD_Instruction(0x0));
              manager.writeInstruction(LOAD_Instruction(0x8 * sizeof(float)));
              // wInstr("LOAD", 0x0);
              // wInstr("LOAD", 0x8 * sizeof(float));
            }
            else if (comm.mac.N == 1)
            {
              manager.writeInstruction(LOAD_Instruction(0x0));
              // wInstr("LOAD", 0x0);
            }
            else
            {
              chprinterr("no");
            }
            loadedKernel = kernel;
          }

          float *dataAddr = addrA + (comm.mac.indexes[0] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize);
          if (comm.mac.indexes[0] <= 0)
          {
            dataAddr = manager.temporaryLoadAddress;
          }
          manager.writeInstruction(LOAD_Instruction((size_t)dataAddr));
          // wInstr("LOAD", dataAddr);
          if (comm.mac.N == 9)
            manager.writeInstruction(LOAD_Instruction((size_t)(dataAddr + 8)));
            // wInstr("LOAD", dataAddr + 8);
          manager.writeInstruction(MAC_Instruction(dataAddr, comm.mac.N, loadedKernel));
          // wInstr("MAC", dataAddr, comm.mac.N, loadedKernel);
        }

        if (c == comm.mac.repeat - 1)
          manager.writeInstruction(RELU_Instruction());
          // wInstr("RELU");
      }
      break;
    }
    case NetCommandType::CLIP:
      // do nothing
      break;
    case NetCommandType::GAP:
    {
      // waiting on lucas
      break;
    }

    default:
      break;
    }
  }
}

void Compiler::compileModel(const Net &net)
{
  float *loadedKernel = 0;
  int count = 0;
  for (const NetCommand &comm : net.commands)
  {
    chprintln("command: ", count++);
    switch (comm.type)
    {
    case NetCommandType::MAC:
    {
      float *addrA = comm.mac.addrA;
      float *addrB = comm.mac.addrB;
      ch_array toWrite = ch_arrcreate(float, 16);
      for (int c = 0; c < comm.mac.repeat; c++)
      {
        for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
        {
          float *kernel = addrB + (comm.mac.indexes[1] + (c % comm.mac.repeatB) * comm.mac.repeatShiftB);
          if (loadedKernel != kernel || c != 0)
          {
            loadedKernel = kernel;
            if (comm.mac.N == 1)
            {
              struct
              {
                float kernel;
                float bias;
                float _zeroPad[2];
              } params;
              memset(&params, 0, sizeof(params));
              params.kernel = addrB[comm.mac.indexes[1]];
              if (c == 0)
                params.bias = comm.mac.addrC ? *comm.mac.addrC : 0;
              else
                params.bias = *(comm.mac.out + shift);
              writeData(loadedKernel, sizeof(params));
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
              if (c == 0)
                params.bias = comm.mac.addrC ? *comm.mac.addrC : 0;
              else
                params.bias = *(comm.mac.out + shift);
              writeData(loadedKernel, sizeof(params));
            }
            else
            {
              chprinterr("unrecognized kernel size");
            }
          }

          memset(toWrite._start, 0, sizeof(float) * 16);
          for (int i = 0; i < comm.mac.N; i++)
          {
            if (comm.mac.indexes[2 * i] == -1)
              ch_arrget(float, toWrite, i) = 0;
            else if (comm.mac.indexes[2 * i] == -2)
              ch_arrget(float, toWrite, i) = 1;
            else
              ch_arrget(float, toWrite, i) = addrA[comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize];
          }
          writeData(toWrite._start, sizeof(float) * (comm.mac.N + (4 - comm.mac.N % 4)));
          // mac
          ++currentPC;
          *(comm.mac.out + shift) = manager.getResult(currentPC);

          if (c == comm.mac.repeat - 1)
          {
            //  relu
            ++currentPC;
            *(comm.mac.out + shift) = manager.getResult(currentPC);
          }
        }
      }
      ch_arrfree(toWrite);
      break;
    }
    case NetCommandType::CLIP:
      // do nothing as this is done in mac
      break;
    case NetCommandType::GAP:
    {
      // waiting on lucas
      break;
    }
    case NetCommandType::GEMM:
    {
      float *addrA = comm.gemm.addrA; // (K,M)
      float *addrB = comm.gemm.addrB; // (N,K)
      float *addrC = comm.gemm.addrC; // (M,N)

      const int K = !comm.gemm.transA ? comm.gemm.dimsA[1] : comm.gemm.dimsA[0];
      const int M = !comm.gemm.transA ? comm.gemm.dimsA[0] : comm.gemm.dimsA[1];
      const int N = !comm.gemm.transB ? comm.gemm.dimsB[1] : comm.gemm.dimsB[0];

      for (int Y = 0; Y < N; Y++)
      {
        for (int X = 0; X < M; X++)
        {
          float val = 0;
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
    default:
      break;
    }
  }
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
int main()
{

#if 0
  Net aModel;
  float a[]={1,2,3,4};
  float b[]={2,3};
  int index[]={0,0,1,1};
    // ,2,0,3,0};
  float out[4];
  NetCommand comm;
  comm.type=NetCommandType::MAC;
  comm.mac.N=1;
  comm.mac.addrA=a;
  comm.mac.addrB=b;
  comm.mac.addrC=0;
  comm.mac.indexes=index;

  comm.mac.repeat = 1;
  comm.mac.repeatShiftA = 0;
  comm.mac.repeatShiftB = 0;

  comm.mac.horShifts = 1;
  comm.mac.horShiftSize = 1;

  comm.mac.vertShift = 2;
  comm.mac.vertShiftSize = 2;
  comm.mac.vertShiftSizeOut = 2;
  comm.mac.out=out;
  aModel.commands.push_back(comm);

  aModel.calculate();

  for(int i=0;i<4;i++)
    chprintln(out[i]);


  return 0;
#endif
  Net model = importModel("../mobilenet-v2-pytorch/mobilenet_v2.onnx");
  chprintln("done");
  int w,h,comp;
  unsigned char *image = stbi_load("/home/chevan/Documents/school/2025-2026/fall term/elec 490/Inference_Accelerator/software/images/DogResize.jpg",
     &w, &h, &comp, STBI_rgb);

  assert(w == h && w == 224 && comp == 3);

  Tensor*input=model.input;
  for (int x = 0; x < 224; x++)
  {
    for (int y = 0; y < 224; y++)
    {
      for (int c = 0; c < 3; c++)
      {
        int arrIndex = input->getIndex(0, c, x, y);
        int imIndex = (c * -1 + 2) + 3 * (y * 224 + x);
        // int imIndex=
        // if(x==0&&y==0)
        //   chprintln(image[imIndex]);
        ch_arrget(float, input->data, arrIndex) = (float)image[imIndex] / 255.0;
        // ch_arrget(float, input->data, arrIndex) = ((float)c+x+y)/(224*224*3);
        // ch_arrget(float, input->data, arrIndex) = 0;
      }
    }
  }
  // ch_arrget(float, input->data, 0) = 1;
  // ch_arrget(float, input->data, 224) = 1;
  // ch_arrget(float, input->data, 224*224) = 1;
  // ch_arrget(float, input->data, 1) = 1;
  // ch_arrget(float, input->data, 2) = 1;

  Compiler compiler=Compiler();
  compiler.writeInstructions(model);
  compiler.compileModel(model);

  // for (int i = 0; i < ch_arrlength(float, model.input->data); i++)
  // {
  //   ch_arrget(float, model.input->data, i) = (float)i / ch_arrlength(float, model.input->data);
  // }
  // return 0;
  // model.calculate();
  
  chprintln("calculated");

  // for(int i=2;i<3;i++)
  // {
  //   const int h=7;
  //   const int w=7;
  //   for(int y=0;y<h;y++)
  //   {
  //     for(int x=0;x<w;x++)
  //     {
  //       float &prob = ch_arrget(float, model.output->data, i*h*w+y*w+x);
  //       chprint(prob,", ");
  //     }
  //     chprintln();
  //   }
  //   chprintln();
  // }
  // for (int i = 0; i < ch_arrlength(float, model.output->data); i++)
  // {
  //   float prob = ch_arrget(float, model.output->data, i);
  //   // if(prob>0.1){
  //     chprintln(i,": ",prob);
  //   // }
  //     // if (i >= 111)
  //     //   break;
  // }
  return 0;





  chprintln(0,": ",ch_arrget(float, model.output->data, 0));
  chprintln(1,": ",ch_arrget(float, model.output->data, 1));
  chprintln(2,": ",ch_arrget(float, model.output->data, 2));
  chprintln(3,": ",ch_arrget(float, model.output->data, 3));
  chprintln(4,": ",ch_arrget(float, model.output->data, 4));
  chprintln(999,": ",ch_arrget(float, model.output->data, 999));
  int maxIndex=0;
  for (int i = 0; i < ch_arrlength(float, model.output->data); i++)
  {
    float prob = ch_arrget(float, model.output->data, i);
    if (prob >= ch_arrget(float, model.output->data, maxIndex))
      maxIndex=i;
  }
  chprintln(maxIndex, ": ", ch_arrget(float, model.output->data, maxIndex));

  // chprintln(ch_arrget(float,model.output->data,263));
  // model.free();

  return 0;
}

