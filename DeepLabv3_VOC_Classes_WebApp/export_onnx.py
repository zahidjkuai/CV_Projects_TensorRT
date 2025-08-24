import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class OnlyOut(torch.nn.Module):
    def __init__(self, m): 
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]

device = "cuda"
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
base = deeplabv3_resnet50(weights=weights).eval().to(device)
model = OnlyOut(base).eval().to(device)

dummy = torch.randn(1, 3, 400, 400, device=device)

torch.onnx.export(
    model,
    dummy,
    "deeplabv3_400_dynamic.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"}
    }
)
print("Saved deeplabv3_400_dynamic.onnx")
