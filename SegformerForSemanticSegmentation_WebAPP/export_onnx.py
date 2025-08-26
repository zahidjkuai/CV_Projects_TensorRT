import torch
from transformers import SegformerForSemanticSegmentation

# Load pretrained SegFormer-B0 on ADE20K
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 512, 512)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "segformer_ade20k.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("ONNX model exported successfully")
