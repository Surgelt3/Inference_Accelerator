import onnx
from onnx import shape_inference

model = onnx.load("mobilenet_v2.onnx")
model = shape_inference.infer_shapes(model)

for node in model.graph.node:
    print(node.op_type, [i for i in node.input], [o for o in node.output])
