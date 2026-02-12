import torch, torch.onnx
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import onnx
from onnx import shape_inference
from PIL import Image, ImageDraw
from onnx import numpy_helper

class ElementwiseLinear(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(ElementwiseLinear, self).__init__()

        # w is the learnable weight of this layer module
        self.w = nn.Parameter(torch.rand(input_size), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # simple elementwise multiplication
        return x*self.w 

model = models.mobilenet_v2(num_classes=1000)

state_dict = torch.load("mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

# print(model)
# summary(model,(3,224,224))


# print(len(model.features)) # 19
# model = nn.Sequential(*[model.features[i] for i in range(19)])
# model=nn.Sequential(*[(model.children())[0],(model.children())[1]])

# classifier = nn.Sequential(*[model.classifier[i] for i in range(2)])

# model = nn.Sequential(*[*[model.features[i] for i in range(19)], model.classifier])

# model = torch.nn.Sequential(*(list(model.children())[:-1]),(list(model.children())[1]))
# model = torch.nn.Sequential(*(list(model.children())[1]))

# list(model.children())[1]=list(model.children())[1][0]
# model = nn.Sequential(*[model.children[i] for i in range(19)])
# model[1]=nn.Sequential(*[model[1].conv[0]])
# model[2]=nn.Sequential(*[model[2].conv[0],model[2].conv[1]])
# modules=[k.split('.') for k, m in model.named_modules() 
#                 if type(m).__name__ == 'Conv2dNormActivation']
# model=nn.Sequential(*modules)
# print(modules)
# for *parent, k in modules:
#     model.get_submodule('.'.join(parent))[int(k)] = nn.Linear(1000)



# print(model)

summary(model,(3,224,224))

torch.onnx.export(model, dummy, "mobilenet-v2-pytorch/mobilenet_v2.onnx", opset_version=13)


onnx_model=onnx.load("mobilenet-v2-pytorch/mobilenet_v2.onnx")
# onnx_model.graph.node.remove(onnx_model.graph.node[-1])
# onnx_model.graph.node.remove(onnx_model.graph.node[-1])
# onnx_model.graph.node[-1].attribute[0].f=0
# onnx_model.graph.node[-1].attribute[1].f=0
# onnx_model.graph.node[-1].attribute[1].f=0
# print(onnx_model.graph.node[-1].attribute)
print(onnx_model.graph.node)
# onnx_model.graph.output.remove(onnx_model.graph.output[-1])
# onnx_model.graph.output.append(onnx.helper.make_tensor_value_info("/GlobalAveragePool_output_0", 1, (1,1280,1,1)))
# onnx_model.graph.output.append(onnx.helper.make_tensor_value_info("/Flatten_output_0", 1, (1,1280,1,1)))

# for node in onnx_model.graph.node:
    # print(node)
    # print()
    # node inputs
    # for idx, node_input_name in enumerate(node.input):
    #     print(idx, node_input_name)
    # node outputs
    # for idx, node_output_name in enumerate(node.output):
    #     print(idx, node_output_name)
        
# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[0]).copy()
# numpy_array[1,:,:,:]=0
# numpy_array[1,0,1,1]=1
# numpy_array[1,0]=[[0,0,0],[0,1,0],[0,0,0]]
# numpy_array[1,1]=[[0,0,0],[0,1,0],[0,0,0]]
# numpy_array[1,2]=[[0,0,0],[0,1,0],[0,0,0]]
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[0].name)
# onnx_model.graph.initializer[0].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[1]).copy()
# # numpy_array[:]=0
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[1].name)
# onnx_model.graph.initializer[1].CopyFrom(tensor)


# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[2]).copy()
# # numpy_array[1,:,:,:]=0
# # numpy_array[1,0]=[[0,0,0],[0,1,0],[0,0,0]]
# # numpy_array[1,0]=[[0,0,0],[0,1,0],[0,0,0]]
# # numpy_array[1,2]=[[0,0,0],[0,1,0],[0,0,0]]
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[2].name)
# onnx_model.graph.initializer[2].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[3]).copy()
# # numpy_array[:]=0
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[3].name)
# onnx_model.graph.initializer[3].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[4]).copy()
# # numpy_array[1,:,:,:]=0
# # numpy_array[1,0]=[[0,0,0],[0,1,0],[0,0,0]]
# # numpy_array[2,0]=[[0,0,0],[0,1,0],[0,0,0]]
# # numpy_array[1,2]=[[0,0,0],[0,1,0],[0,0,0]]
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[4].name)
# onnx_model.graph.initializer[4].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[5]).copy()
# # numpy_array[:]=0
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[5].name)
# onnx_model.graph.initializer[5].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[6]).copy()
# # numpy_array[:]=0
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[6].name)
# onnx_model.graph.initializer[6].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[7]).copy()
# # numpy_array[:]=0
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[7].name)
# onnx_model.graph.initializer[7].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[8]).copy()
# # numpy_array[2,:,:,:]=0
# # numpy_array[2,0]=[[0,0,0],[0,1,0],[0,0,0]]
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[8].name)
# onnx_model.graph.initializer[8].CopyFrom(tensor)

# numpy_array=numpy_helper.to_array(onnx_model.graph.initializer[9]).copy()
# # numpy_array[:]=0
# tensor = numpy_helper.from_array(numpy_array,onnx_model.graph.initializer[9].name)
# onnx_model.graph.initializer[9].CopyFrom(tensor)



onnx.checker.check_model(onnx_model)
onnx.save(onnx_model,"mobilenet-v2-pytorch/mobilenet_v2.onnx")


# inferred_model = shape_inference.infer_shapes(onnx_model)
# print(inferred_model.graph.value_info)
