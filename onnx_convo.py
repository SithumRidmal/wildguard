import torch
from torchvision import models

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("MobileNetV2 ONNX exported successfully")
