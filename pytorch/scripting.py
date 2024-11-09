import torch
from transformers import MarianTokenizer

# Load the tokenizer and quantized model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer = MarianTokenizer.from_pretrained(model_name)
quantized_model = torch.load("../models/quantized_marianmt.pth")  # Load the full model object
# Trace or script the quantized model
scripted_model = torch.jit.script(quantized_model)
scripted_model.save("../models/quantized_marianmt_scripted.pt")
# Load the TorchScript model
scripted_model = torch.jit.load("../models/quantized_marianmt_scripted.pt")
scripted_model.eval()

# Run inference
with torch.no_grad():
    translated_tokens = scripted_model.generate(**inputs)

