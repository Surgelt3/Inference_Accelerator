import torch, torch.onnx
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

model = models.mobilenet_v2(num_classes=1000)

state_dict = torch.load("mobilenet_v2-b0353104.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

summary(model,(3,224,224))

torch.onnx.export(model, dummy, "mobilenet_v2.onnx", opset_version=13)
