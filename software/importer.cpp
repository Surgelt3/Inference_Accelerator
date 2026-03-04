#include "importer.hpp"
#include <utils.hpp>
#include <fstream>
#include <math.h>

#include <onnx.pb.h>


static Tensor parseTensor(onnx::TensorProto t)
{
  size_t size = 1;
  ch_array dim;
  if (t.dims_size() == 0)
  {
    size = t.float_data_size() == 0 ? t.raw_data().size() / 4 : t.float_data_size();
    dim = ch_arrcreate(int, 4);
    for(int i=0;i<4;i++)
      ch_arrget(int, dim, i) = 0;
    ch_arrget(int, dim, 0) = size;
  }
  else
  {
    dim=ch_arrcreate(int, t.dims_size());
    for (int i = 0; i < t.dims_size(); i++)
    {
      if (t.dims(i) < 0)
        chprinterr("size < 0\n");
      size *= t.dims(i);
      ch_arrget(int, dim, i) = t.dims(i);
    }
  }

  ch_array a = ch_arrcreate(float, size);

  // load data type shit
  if (t.data_location() == onnx::TensorProto::DEFAULT)
  {
    switch (t.data_type())
    {
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::COMPLEX64:
      if (t.float_data_size() != size || size == 0)
      {
        if (t.raw_data().size() / 4 != size)
          chstop("TODO");

        for (int i = 0; i < size; i++)
        {
          union {
            float f;
            int32_t i;
          }d;
          memcpy(&d.i, t.raw_data().data() + 4 * i,sizeof(int32_t));
          #if BYTE_ORDER == BIG_ENDIAN
            d.i=__bswap_32(d.i);
          #endif
          ch_arrget(float, a, i) = d.f;
        }
      }
      else
        memcpy(a._start, t.raw_data().c_str(), size * sizeof(float));
      break;
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT4:
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT4:
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::FLOAT8E4M3FN:
    case onnx::TensorProto::FLOAT8E4M3FNUZ:
    case onnx::TensorProto::FLOAT8E5M2:
    case onnx::TensorProto::FLOAT8E5M2FNUZ:
    case onnx::TensorProto::FLOAT8E8M0:
    case onnx::TensorProto::FLOAT4E2M1:
      chstop("TOOD");
      break;
    case onnx::TensorProto::STRING:
      chstop("TOOD");
      break;
    case onnx::TensorProto::INT64:
    case onnx::TensorProto::UINT64:
      chstop("TOOD");
      break;
    case onnx::TensorProto::DOUBLE:
    case onnx::TensorProto::COMPLEX128:
      chstop("TOOD");
      break;
    case onnx::TensorProto::UNDEFINED:
    default:
      chstop("Undefined data type");
      break;
    }
  }
  else
  {
    chstop("TODO");
  }
  return Tensor(a,dim);
}

Net importModel(std::string path)
{
  onnx::ModelProto model;
  std::fstream input(path,
                     std::ios::in | std::ios::binary);
  model.ParseFromIstream(&input);

  // hash map of all arrays of each layer
  ch_hash arrayNameMap = ch_hashcreatesize(Tensor *, model.graph().node_size() * 4);
  // complete model
  Net aModel = Net();

  aModel.input_values.resize(1+model.graph().initializer_size());
  // create input array
  if (model.graph().input(0).has_name())
  {
    int size = 1;
    ch_array dim = ch_arrcreate(int, model.graph().input(0).type().tensor_type().shape().dim_size());
    for (int i = 0; i < model.graph().input(0).type().tensor_type().shape().dim_size(); i++)
    {
      size *= model.graph().input(0).type().tensor_type().shape().dim(i).dim_value();
      ch_arrget(int, dim, i) = model.graph().input(0).type().tensor_type().shape().dim(i).dim_value();
    }
    ch_array a = ch_arrcreate(float, size);
    aModel.input_values[0] = Tensor(a, dim);
    aModel.input = &aModel.input_values[0];
    ch_hashinsert(Tensor *, arrayNameMap, model.graph().input(0).name().c_str(), &aModel.input_values[0]);
  }

  // store arrays in the model and hashmap
  for (int i = 0; i < model.graph().initializer_size(); i++)
  {
    aModel.input_values[1+i] = parseTensor(model.graph().initializer(i));
    ch_hashinsert(Tensor *, arrayNameMap, model.graph().initializer(i).name().c_str(), &aModel.input_values[1+i]);
  }

  aModel.layers.reserve(model.graph().node_size());
  for (int i = 0; i < model.graph().node_size(); i++)
  {
    aModel.layers.push_back(Layer());
    Layer &layer = aModel.layers.back();

    const onnx::NodeProto &node = model.graph().node(i);
    chprintln("name ", i, ": ", node.name());
    chprintln("\top: ", node.op_type());

    for (int j = 0; j < node.input().size(); j++)
    {
      Tensor *a = ch_hashget(Tensor *, arrayNameMap, node.input().Get(j).c_str());
      if (a == ch_hash_NOTFOUND)
      {
        chprinterr("Unable to find layer: %s\n", node.input().Get(j).c_str());
      }
      layer.layer_input.push_back(a);
    }

    chassert(node.output().size() == 1, "Layer must have a single output array");
    ch_hashinsert(Tensor *, arrayNameMap, node.output().Get(0).c_str(), &layer.layer_output);

    // specifies stuff like convolution size and shit
    struct CommandAttributes
    {
      std::string auto_pad = "NOTSET";
      int dilations[4] = {1, 1, 1, 1};
      int group = 1;
      int kernel_shape[2] = {0, 0};
      int pads[4] = {0, 0, 0, 0};
      int strides[2] = {1, 1};
      // sparse_value : sparse_tensor
      Tensor value;
      float value_float;
      float value_floats[8];
      int value_int;
      int value_ints[4];
      std::string value_string;
      std::string value_strings[4];
      int axis = 1;
      float alpha = 1;
      float beta = 1;
      int transA = 0;
      int transB = 0;
    } attributes;
    for (int j = 0; j < node.attribute_size(); j++)
    {
      const std::string &attName = node.attribute(j).name();
      if (attName == "auto_pad")
      {
        chstop("TODO");
      }
      else if (attName == "dilations")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS)
          chprinterr("%s is the wrong type\n", attName.c_str());
        for (int k = 0; k < node.attribute(j).ints_size(); k++)
        {
          attributes.dilations[k] = node.attribute(j).ints(k);
          assert(attributes.dilations[k] == 1);
        }
      }
      else if (attName == "group")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.group = node.attribute(j).i();
      }
      else if (attName == "kernel_shape")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS)
          chprinterr("%s is the wrong type\n", attName.c_str());
        for (int k = 0; k < node.attribute(j).ints_size(); k++)
          attributes.kernel_shape[k] = node.attribute(j).ints(k);
      }
      else if (attName == "pads")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS)
          chprinterr("%s is the wrong type\n", attName.c_str());
        for (int k = 0; k < node.attribute(j).ints_size(); k++)
        {
          attributes.pads[k] = node.attribute(j).ints(k);
          assert(attributes.pads[k]<=1);
        }
      }
      else if (attName == "strides")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS)
          chprinterr("%s is the wrong type\n", attName.c_str());
        for (int k = 0; k < node.attribute(j).ints_size(); k++)
        {
          attributes.strides[k] = node.attribute(j).ints(k);
        }
      }
      else if (attName == "value")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.value = parseTensor(node.attribute(j).t());
      }
      else if (attName == "axis")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.axis = node.attribute(j).i();
      }
      else if (attName == "alpha")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.alpha = node.attribute(j).f();
      }
      else if (attName == "beta")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.beta = node.attribute(j).f();
      }
      else if (attName == "transA")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.transA = node.attribute(j).i();
      }
      else if (attName == "transB")
      {
        if (node.attribute(j).type() != onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT)
          chprinterr("%s is the wrong type\n", attName.c_str());
        attributes.transB = node.attribute(j).i();
      }
      else
      {
        chprinterr("TODO: %s\n", attName.c_str());
      }
    }

    if (node.op_type() == "Conv")
    {
      const Tensor &base = *layer.layer_input[0];
      const Tensor &kernel = *layer.layer_input[1];
      const Tensor &bias = *layer.layer_input[2];
      int *temporaryIndexStore = (int *)malloc(sizeof(int) * 2 * kernel.width() * kernel.height());
      layer.layer_output.dim = ch_arrcreate(int, 4);
      layer.layer_output.batch() = 1;
      layer.layer_output.channel() = kernel.batch();
      layer.layer_output.width() = (base.width() - attributes.kernel_shape[0] + attributes.pads[0] + attributes.pads[1]) / attributes.strides[0] + 1;
      layer.layer_output.height()= (base.height() - attributes.kernel_shape[1] + attributes.pads[2] + attributes.pads[3]) / attributes.strides[1] + 1;
      layer.layer_output.data = ch_arrcreate(float, layer.layer_output.channel() * layer.layer_output.width() * layer.layer_output.height());

      ch_array cmdStack = ch_arrstack(NetCommand, layer.layer_output.batch() * layer.layer_output.channel() * layer.layer_output.width() * layer.layer_output.height());
      for (uint z = 0; z < layer.layer_output.channel(); z++)
      {
        int outY = 0;
        int shiftY = 0;
        for (int y = -attributes.pads[1]; y <= base.height() - attributes.kernel_shape[1] + attributes.pads[3]; y += attributes.strides[1] + shiftY, outY++)
        {
          int outX=0;
          bool previousCommandRepeat = false;
          for (int x = -attributes.pads[0]; x <= base.width() - attributes.kernel_shape[0] + attributes.pads[2]; x += attributes.strides[0], outX++)
          {
            NetCommand comm;
            comm.type = MAC;
            comm.mac.N = kernel.width() * kernel.height();
            comm.mac.repeat = base.channel() / attributes.group;
            comm.mac.repeatB = kernel.channel();
            comm.mac.repeatShiftA = base.getIndex(0, 1, 0, 0) - base.getIndex(0, 0, 0, 0);
            comm.mac.repeatShiftB = kernel.getIndex(0, 1, 0, 0) - base.getIndex(0, 0, 0, 0);
            comm.mac.horShifts = 0;
            comm.mac.horShiftSize = base.getIndex(0, 0, attributes.strides[0], 0) - base.getIndex(0, 0, 0, 0);
            comm.mac.addrA = (float *)base.data._start;
            comm.mac.addrB = (float *)kernel.data._start;
            comm.mac.addrC = ((float *)bias.data._start) + z;
            comm.mac.out = layer.layer_output.get(0, z, outX, outY);
            comm.mac.indexes = NULL;
            comm.mac.vertShift = 1;
            comm.mac.vertShiftSize = 0;
            comm.mac.vertShiftSizeOut = layer.layer_output.getIndex(0, 0, 0, 1) - layer.layer_output.getIndex(0, 0, 0, 0);

            bool canRepeatHor = true;
            bool canRepeatVert = true;
            // the entire column is OOB
            bool vertIsOOB[3] = {true, true, true}; // TODO this is bad, please make variable size
            for (int dy = 0; dy < attributes.kernel_shape[1]; dy++)
            {
              // the entire row is OOB
              bool horIsOOB = true;
              for (int dx = 0; dx < attributes.kernel_shape[0]; dx++)
              {
                int aIndex = 0;
                if (x + dx < 0 || y + dy < 0 || x + dx >= base.width() || y + dy >= base.height())
                {
                  aIndex = -1;
                }
                else
                {
                  if (attributes.group != 1)
                    aIndex = base.getIndex(0, z, x + dx, y + dy);
                  else
                    aIndex = base.getIndex(0, 0, x + dx, y + dy);
                  vertIsOOB[dx] = false;
                  horIsOOB = false;
                }
                const int bIndex = kernel.getIndex(z, 0, dx, dy);
                assert(!isnan(ch_arrget(float,kernel.data,bIndex)));
                temporaryIndexStore[0 + 2 * (attributes.kernel_shape[1] * dy + dx)] = aIndex;
                temporaryIndexStore[1 + 2 * (attributes.kernel_shape[1] * dy + dx)] = bIndex;
              }
              canRepeatVert = canRepeatVert && !horIsOOB;
              canRepeatVert=false;
            }
            for (int dx = 0; dx < attributes.kernel_shape[0]; dx++)
            {
              canRepeatHor = canRepeatHor && !vertIsOOB[dx];
            }
            if (canRepeatVert)
            {
              comm.mac.vertShift = layer.layer_output.height()-2;
              comm.mac.vertShiftSize = base.getIndex(0, 0, 0, attributes.strides[1]) - base.getIndex(0, 0, 0, 0);
              shiftY = comm.mac.vertShift * attributes.strides[1];
            }
            else
            {
              shiftY = 0;
            }
            if (canRepeatHor && previousCommandRepeat)
            {
              ((NetCommand *)cmdStack._end - 1)->mac.horShifts++;
            }
            else
            {
              comm.mac.indexes = (int *)malloc(sizeof(int) * 2 * kernel.width() * kernel.height());
              memcpy(comm.mac.indexes, temporaryIndexStore, sizeof(int) * 2 * kernel.width() * kernel.height());
              ch_arrpush(NetCommand, cmdStack, comm) else
              {
                chprinterr("stack full\n");
              }
            }
            previousCommandRepeat = canRepeatHor;
          }
        }
      }
      for(int i=0;i<ch_arrlength(NetCommand,cmdStack);i++)
      {
        aModel.commands.push_back(ch_arrget(NetCommand, cmdStack, i));
      }
      ch_arrfree(cmdStack);
    }
    else if (node.op_type() == "Constant")
    {
      if (attributes.value.batch() < 0 || attributes.value.channel() < 0 || attributes.value.width() < 0 || attributes.value.height() < 0)
      {
        chprinterr("size < 0\n");
      }
      if(attributes.value.data._start)
      {
        layer.layer_output = attributes.value;
      }
      else
      {
        chprinterr("constant type unimplemented");
      }
    }
    else if (node.op_type() == "Clip")
    {
      const Tensor &base = *layer.layer_input[0];
      const Tensor &min = *layer.layer_input[1];
      const Tensor &max = *layer.layer_input[2];
      layer.layer_output.dim = ch_arrcopy(base.dim);
      layer.layer_output.data = ch_arrcopy(base.data);
      NetCommand comm;
      comm.type = CLIP;
      comm.clip.N = ch_arrlength(float, base.data);
      comm.clip.addrA = (float *)base.data._start;
      comm.clip.addrMin = (float *)min.data._start;
      comm.clip.addrMax = (float *)max.data._start;
      comm.clip.out = (float *)layer.layer_output.data._start;
      aModel.commands.push_back(comm);
    }
    else if (node.op_type() == "Add")
    {
      const Tensor &tensorA = *layer.layer_input[0];
      const Tensor &tensorB = *layer.layer_input[1];
      layer.layer_output.dim = ch_arrcopy(tensorA.dim);
      layer.layer_output.data = ch_arrcopy(tensorA.data);
      NetCommand comm;
      comm.type = ADD;
      comm.add.N = ch_arrlength(float, tensorA.data);
      comm.add.addrA = (float *)tensorA.data._start;
      comm.add.addrB = (float *)tensorB.data._start;
      comm.add.out = (float *)layer.layer_output.data._start;
      aModel.commands.push_back(comm);
    }
    else if (node.op_type() == "GlobalAveragePool")
    {
      const Tensor &base = *layer.layer_input[0];
      layer.layer_output.dim = ch_arrcopy(base.dim);
      ch_arrget(int, layer.layer_output.dim, 2) = 1;
      ch_arrget(int, layer.layer_output.dim, 3) = 1;
      layer.layer_output.data = ch_arrcreate(int, layer.layer_output.channel() * layer.layer_output.channel());

      for (int j = 0; j < base.batch(); j++)
      {
        for (int k = 0; k < base.channel(); k++)
        {
          NetCommand comm;
          comm.type = GAP;
          comm.mac.repeat = 1;
          comm.mac.repeatB = 1;
          comm.mac.repeatShiftA = 0;
          comm.mac.repeatShiftB = 0;
          comm.mac.N = base.width() * base.height();
          comm.mac.horShifts = 0;
          comm.mac.vertShift = 1;
          comm.mac.addrA = (float *)base.data._start;
          comm.mac.addrB = (float *)base.data._start;
          comm.mac.addrC = NULL;
          comm.mac.out = ((float *)layer.layer_output.data._start) + k + layer.layer_output.channel() * j;
          comm.mac.indexes = (int *)malloc(sizeof(int) * 2 * base.width() * base.height());

          for (int x = 0; x < base.width(); x++)
          {
            for (int y = 0; y < base.height(); y++)
            {
              comm.mac.indexes[2 * (x + y * base.width())] = base.getIndex(j,k,x,y);
              comm.mac.indexes[2 * (x + y * base.width()) + 1] = -2;
            }
          }
          aModel.commands.push_back(comm);
        }
      }
    }
    else if (node.op_type() == "Flatten")
    {
      const Tensor &base = *layer.layer_input[0];
      layer.layer_output.dim = ch_arrcopy(base.dim);
      layer.layer_output.batch() = 1;
      layer.layer_output.channel() = base.batch() * base.channel() * base.width() * base.height();
      layer.layer_output.width() = 1;
      layer.layer_output.height() = 1;
      layer.layer_output.data = base.data;
    }
    else if (node.op_type() == "Gemm")
    {
      const Tensor &tensorA = *layer.layer_input[0];
      const Tensor &tensorB = *layer.layer_input[1];
      const Tensor &tensorC = *layer.layer_input[2];
      layer.layer_output.dim = ch_arrcopy(tensorC.dim);

      const int M = !attributes.transA ? tensorA.batch() : tensorA.channel();
      const int N = !attributes.transB ? tensorB.channel() : tensorB.batch();

      layer.layer_output.batch() = M;
      layer.layer_output.channel() = N;
      layer.layer_output.width() = 1;
      layer.layer_output.height() = 1;
      layer.layer_output.data = ch_arrcreate(float, M *N);

      NetCommand comm;
      comm.type=GEMM;
      comm.gemm.alpha=attributes.alpha;
      comm.gemm.beta=attributes.beta;
      comm.gemm.transA=attributes.transA;
      comm.gemm.transB=attributes.transB;
      comm.gemm.addrA = (float *)tensorA.data._start;
      comm.gemm.addrB = (float *)tensorB.data._start;
      comm.gemm.addrC = (float *)tensorC.data._start;
      comm.gemm.out = (float *)layer.layer_output.data._start;
      comm.gemm.dimsA = (int *)tensorA.dim._start;
      comm.gemm.dimsB = (int *)tensorB.dim._start;
      comm.gemm.dimsC = (int *)tensorC.dim._start;

      aModel.commands.push_back(comm);
    }
    else
    {
      chprinterr("unimplemented layer type %s\n", node.op_type().c_str());
    }
    chprintln("\tout size: ", layer.layer_output.batch(), ",", layer.layer_output.channel(), ",", layer.layer_output.width(), ",", layer.layer_output.height());
    chprintln("\tcommands: ",aModel.commands.size());
  }
  aModel.output = ch_hashget(Tensor *, arrayNameMap, model.graph().output(0).name().c_str());
  chassert(aModel.output != ch_hash_NOTFOUND, "Output array not found");

  aModel.commands.shrink_to_fit();
  ch_hashfree(arrayNameMap);

  return aModel;
}
