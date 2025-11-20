import torch, torch.onnx
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import onnx
from onnx import shape_inference
from PIL import Image, ImageDraw

model = models.mobilenet_v2(num_classes=1000)

state_dict = torch.load("mobilenet-v2-pytorch/mobilenet_v2-b0353104.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

# summary(model,(3,224,224))
# summary(model,(3,224,224))

# torch.onnx.export(model, dummy, "mobilenet-v2-pytorch/mobilenet_v2.onnx", opset_version=13)


onnx_model=onnx.load("mobilenet-v2-pytorch/mobilenet_v2.onnx")
onnx.checker.check_model(onnx_model)


# inferred_model = shape_inference.infer_shapes(onnx_model)
# print(inferred_model.graph.value_info)
