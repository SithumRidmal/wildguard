import torch
from torchvision.models import vgg16, VGG16_Weights

model = vgg16(weights=VGG16_Weights.DEFAULT)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "vgg16.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11

    
)
