import torch
from transformers import MarianMTModel

# Load and quantize the model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Replace with your MarianMT model
model = MarianMTModel.from_pretrained(model_name)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

import torch.nn.utils.prune as prune

for module in quantized_model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.4)

# Save the full quantized model object (not just the state dict)
torch.save(quantized_model, "pruned_quantized_marianmt.pth")
print("Quantized model saved as 'quantized_marianmt.pth'.")
