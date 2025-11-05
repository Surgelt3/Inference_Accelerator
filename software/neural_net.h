#pragma once

#include <chevan_utils_min.h>
#include <chevan_utils_array.h>
#include <vector>
#include <string>

struct Tensor
{
  ch_array data;
  ch_array dim;
  Tensor()
  {
    data = {0, 0, 0};
    dim = {0, 0, 0};
  }
  Tensor(ch_array data, ch_array dim)
  {
    this->data = data;
    this->dim = dim;
  }
  int &batch() const
  {
    return ch_arrget(int, dim, 0);
  }
  int &channel() const
  {
    return ch_arrget(int, dim, 1);
  }
  int &width() const
  {
    return ch_arrget(int, dim, 2);
  }
  int &height() const
  {
    return ch_arrget(int, dim, 3);
  }
  void free()
  {
    if(data._start)
      ch_arrfree(data);
    if(dim._start)
      ch_arrfree(dim);
  }
};

// structure storing a single layer
struct Layer
{
  std::vector<Tensor *> layer_input;
  Tensor layer_output;
  Layer()
  {
    layer_input = std::vector<Tensor *>();
    layer_output = Tensor();
  }
  void free()
  {
    layer_output.free();
  }
};

enum NetCommandType{
  MAC,
  LOAD,
  CLIP,
  ADD,
  ADDN,
  GAP,
  FLAT,
  GEMM
};
struct NetCommand
{
  NetCommandType type;
  union
  {
    struct
    {
      int N;
      float *addrA, *addrB, *addrC, *out;
      int *indexes;
    } mac;
    struct
    {
      int N;
      float *addrA, *addrMin, *addrMax,*out;
    } clip;
    struct
    {
      int N;
      float *addrA, *addrB,*out;
    } add;

    struct
    {
      int N;
      float *addr,*out;
      int*indexes;
    }addn;
    struct
    {

    } gemm;
  };

  std::string toString()
  {
    std::string res;
    switch (type)
    {
    case NetCommandType::MAC:
      char res2[64];
      sprintf(res2, "MAC %u %p %p", mac.N, mac.addrA, mac.addrB);
      res = res2;
      // for (int i : mac.indexes)
      // {
      //   res += " ";
      //   res += i;
      // }
      return res;
    default:
      break;
    }
    return "";
  }
};

// object storing the entire model and command to run it
struct Net
{
  // each layer in the model
  std::vector<Layer> layers;
  // input vectors 
  std::vector<Tensor> input_values;
  Tensor *input;
  Tensor *output;
  /*
  structure of commands
  index -1 is constant 0
  index -2 is constant 1
    MAC N,addrA,addrB,addrC,x,a,y,b,z,c,x1,a1,y1,b1,z1,c1 -> addrC[0]+addrA[x]*addrB[a]+addrA[y]*addrB[b]+addrA[z]*addrB[c]+...,addrA[x1]*addrB[a1]+addrA[y1]*addrB[b1]+addrA[z1]*addrB[c1]
    LOAD
    CLIP N, addrA, addrMin, addrMax
    ADD N, addrA, addrB
    APP N,addr -> act(addr[0]), act(addr[1]), ...
  */
  std::vector<NetCommand> commands;
  Net()
  {
    layers = std::vector<Layer>();
    input_values = std::vector<Tensor>();
    input = 0;
    output = 0;
  }
  void free()
  {
    for (int i = 0; i < input_values.size(); i++)
      input_values[i].free();
    for (int i = 0; i < layers.size(); i++)
      layers[i].free();
  }
};
