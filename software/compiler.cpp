#include "compiler.hpp"
#include "importer.hpp"

#if 1
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

  
  assert(w==h&&w==224);

  Tensor*input=model.input;
  for (int x = 0; x < 224; x++)
  {
    for (int y = 0; y < 224; y++)
    {
      for (int c = 0; c < 3; c++)
      {
        int arrIndex = input->getIndex(0, c, x, y);
        int imIndex = c + 3 * (y * 224 + x);
        // ch_arrget(float, input->data, arrIndex) = (float)image[imIndex] / 255.0;
        ch_arrget(float, input->data, arrIndex) = ((float)c+x+y)/(224*224*3);
        // ch_arrget(float, input->data, arrIndex) = 0;
      }
    }
  }
  // ch_arrget(float, input->data, 0) = 1;
  // ch_arrget(float, input->data, 224) = 1;
  // ch_arrget(float, input->data, 224*224) = 1;
  // ch_arrget(float, input->data, 1) = 1;
  // ch_arrget(float, input->data, 2) = 1;

  // compiler.compileModel(model);

  // for (int i = 0; i < ch_arrlength(float, model.input->data); i++)
  // {
  //   ch_arrget(float, model.input->data, i) = (float)i / ch_arrlength(float, model.input->data);
  // }
  // return 0;
  model.calculate();
  chprintln("calculated");

  for(int i=2;i<3;i++)
  {
    const int h=7;
    const int w=7;
    for(int y=0;y<h;y++)
    {
      for(int x=0;x<w;x++)
      {
        float &prob = ch_arrget(float, model.output->data, i*h*w+y*w+x);
        chprint(prob,", ");
      }
      chprintln();
    }
    chprintln();
  }
  // for (int i = 0; i < ch_arrlength(float, model.output->data); i++)
  // {
  //   float prob = ch_arrget(float, model.output->data, i);
  //   // if(prob>0.1){
  //     chprintln(i,": ",prob);
  //   // }
  //     // if (i >= 111)
  //     //   break;
  // }
  chprintln(ch_arrget(float, model.output->data, 0));
  chprintln(ch_arrget(float, model.output->data, 1));
  chprintln(ch_arrget(float, model.output->data, 2));
  chprintln(ch_arrget(float, model.output->data, 3));
  chprintln(ch_arrget(float, model.output->data, 4));
  chprintln(ch_arrget(float, model.output->data, 5));

  // chprintln(ch_arrget(float,model.output->data,263));
  // model.free();

  return 0;
}
#else
int main(){
  uchar *VIRT_MEM = (uchar *)malloc(0x5000);
  float *readPtr = (float*)(VIRT_MEM + 512 + sizeof(float) * 2);
  MemManager manager = MemManager(VIRT_MEM);

  float data[]={1,2,3};
  float data1[12][1]={{420.69},{4},{5}};
  manager.schedule(data,sizeof(float)*3);
  readPtr = (float *)manager.use(data, sizeof(int) * 3);
  chprintln(readPtr);
  chprintln(*manager.constant0, " ", *manager.constant1);
  chprintln(readPtr[0], " ", readPtr[1], " ", readPtr[2]);
  ((uint32_t *)readPtr)[-1] &= ~0x1;
  for (int i = 0; i < 12; i++)
  {
    manager.schedule(data1[i], sizeof(float) * 1);
  }

  readPtr=(float*)manager.request(data, sizeof(float) * 3);
  chprintln(readPtr);
  chprintln(*manager.constant0," ",*manager.constant1);
  chprintln(readPtr[0]," ",readPtr[1]," ",readPtr[2]);

  readPtr=(float*)manager.request(data1[0],sizeof(float));
  chprintln(readPtr[0]);

  return 0;
}
#endif

