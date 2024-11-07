import torch
from transformers import MarianMTModel

# Load your MarianMT model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Replace with your MarianMT model name
model = MarianMTModel.from_pretrained(model_name)

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
quantized_model.save_pretrained("quantized_marianmt")

