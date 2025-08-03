import torch
import torch.nn as nn
import torchvision

class ResNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

    def forward(self, input):  # input tensor named 'input'
        return self.model(input)

# Prepare model
model = ResNetWrapper()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with dynamic batch size on input/output dim 0
torch.onnx.export(
    model,
    dummy_input,
    "/Tensorflow_Vision/Resnet18/resnet18_dynamicinput.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},   # make batch dim dynamic
        "output": {0: "batch_size"}   # output batch dim dynamic too
    },
    opset_version=11,
    do_constant_folding=True,

)