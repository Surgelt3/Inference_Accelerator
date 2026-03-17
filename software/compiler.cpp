#include "compiler.hpp"
#include "importer.hpp"

static uint32_t currentPC = 0;
static size_t maxDataAddr=(1<<20)*5;
static const int LOAD_SIZE=8;
void Compiler::writeInstructions(const Net &net)
{
  for (const NetCommand &comm : net.commands)
  {
    switch (comm.type)
    {
    case NetCommandType::MAC:
    {
      for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
      {
        int l = comm.mac.repeat * comm.mac.N;
        if (shift == 0)
        {
          for (int i = 0; i < l / LOAD_SIZE; i++)
          {
            manager.writeInstruction(LOAD_Instruction(0x0 + i * LOAD_SIZE * 0x4));
          }
        }
        float *dataAddr;
        dataAddr = (float *)((manager.PC * 4 * 16) % (maxDataAddr - 4 * 16) + (4 * 16));
        for (int i = 0; i < l / LOAD_SIZE; i++)
        {
          manager.writeInstruction(LOAD_Instruction((size_t)dataAddr + i * LOAD_SIZE * 0x4));
        }
        manager.writeInstruction(MAC_Instruction(dataAddr, comm.mac.N, 0x0));
      }
      break;
    }
    default:
      break;
    }
  }
}

void Compiler::compileModel(Net &net)
{
  int count = 0;
  for (const NetCommand &comm : net.commands)
  {
    // chprintln("command: ", count++);
    switch (comm.type)
    {
    case MAC:
    {
      float *addrA = comm.mac.addrA;
      float *addrB = comm.mac.addrB;
      ch_array toWrite = ch_arrstack(float, comm.mac.N *comm.mac.repeat + LOAD_SIZE);
      ch_array kernelToWrite = ch_arrstack(float, comm.mac.N *comm.mac.repeat + 1 + LOAD_SIZE);
      for (int shift = 0; shift < comm.mac.horShifts + 1; shift++)
      {
        toWrite._end = toWrite._start;
        if (shift == 0)
          kernelToWrite._end = kernelToWrite._start;
        for (int c = 0; c < comm.mac.repeat; c++)
        {
          for (int i = 0; i < comm.mac.N; i++)
          {
            float valA;
            if (comm.mac.indexes[2 * i] == -1)
              valA = 0;
            else if (comm.mac.indexes[2 * i] == -2)
              valA = 1;
            else
              valA = addrA[comm.mac.indexes[2 * i] + c * comm.mac.repeatShiftA + shift * comm.mac.horShiftSize];
            ch_arrpush(float, toWrite, valA);
            if (shift == 0)
            {
              float valB;
              if (comm.mac.indexes[2 * i + 1] == -1)
                valB = 0;
              else if (comm.mac.indexes[2 * i + 1] == -2)
                valB = 1;
              else
                valB = addrB[comm.mac.indexes[2 * i + 1] + (c % comm.mac.repeatB) * comm.mac.repeatShiftB];
              ch_arrpush(float, kernelToWrite, valB);
            }
          }
        }

        for (int j = 0; j < LOAD_SIZE - ch_arrlength(float, toWrite) % LOAD_SIZE; j++)
        {
          ch_arrpush(float, toWrite, 0);
        }
        if (shift == 0)
        {
          if (comm.mac.addrC)
          {
            ch_arrpush(float, kernelToWrite, *comm.mac.addrC);
          }
          else
          {
            ch_arrpush(float, kernelToWrite, 0);
          }
          for (int j = 0; j < LOAD_SIZE - ch_arrlength(float, kernelToWrite) % LOAD_SIZE; j++)
          {
            ch_arrpush(float, kernelToWrite, 0);
          }
          ++currentPC;
          manager.writeData((float *)kernelToWrite._start, sizeof(float) * ch_arrlength(float, kernelToWrite));
        }
        ++currentPC;
        manager.writeData((float *)toWrite._start, sizeof(float) * ch_arrlength(float,toWrite));
        
        // mac operation        
        ++currentPC;
#if ONDEVICE
        *(comm.mac.out + shift) = manager.getResult(currentPC);
#else
        float sum = 0;
        for (int i = 0; i < comm.mac.N * comm.mac.repeat; i++)
        {
          sum += ch_arrget(float, kernelToWrite, i) * ch_arrget(float, toWrite, i);
        }
        float bias = ch_arrget(float, kernelToWrite, comm.mac.N *comm.mac.repeat);
        *(comm.mac.out + shift) = sum + bias;
#endif
      }
      ch_arrfree(kernelToWrite);
      ch_arrfree(toWrite);
      break;
    }

    case NetCommandType::CLIP:
    {
      assert(*comm.clip.addrMax == 6);
      assert(*comm.clip.addrMin == 0);
      net.useCommand(comm);
      break;
    }
    case NetCommandType::ADD:
    case NetCommandType::GAP:
    case NetCommandType::GEMM:
    case NetCommandType::COPY:
    {
      net.useCommand(comm);
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

  chprintln("Start!");
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
  chprintln("done import");


  while(true)
  {
    std::string imageIndex = "";
    do
    {
      chprintln("Enter image (1: apple, 2:strawberry, 3: orange)");
      std::cin >> imageIndex;
    } while (imageIndex != "1" && imageIndex != "2" && imageIndex != "3");

    std::string paths[] = {"./apple.jpg", "./strawberry.jpg", "./orange.jpg"};

    int w, h, comp;
    unsigned char *image = stbi_load(paths[stoi(imageIndex)-1].c_str(),
                                     &w, &h, &comp, STBI_rgb);

    assert(w == h && w == 224 && comp == 3);

    Tensor *input = model.input;
    for (int x = 0; x < 224; x++)
    {
      for (int y = 0; y < 224; y++)
      {
        for (int c = 0; c < 3; c++)
        {
          int arrIndex = input->getIndex(0, c, x, y);
          int imIndex = (c * -1 + 2) + 3 * (y * 224 + x);
          ch_arrget(float, input->data, arrIndex) = (float)image[imIndex] / 255.0;
          // ch_arrget(float, input->data, arrIndex) = 0;
        }
      }
    }

    Compiler compiler = Compiler();
    // compiler.writeInstructions(model);
    // compiler.compileModel(model);

    model.calculate();

    chprintln("calculated");
    int maxIndex = 0;
    for (int i = 0; i < ch_arrlength(float, model.output->data); i++)
    {
      float prob = ch_arrget(float, model.output->data, i);
      if (prob >= ch_arrget(float, model.output->data, maxIndex))
        maxIndex = i;
    }
    // chprintln(maxIndex, ": ", ch_arrget(float, model.output->data, maxIndex));

    if (maxIndex == 948)
    {
      chprintln("Apple");
    }
    else
    {
      chprintln("maxIndex: ", maxIndex);
    }
  }

  model.free();
  return 0;
}

