import torch
from transformers import MarianTokenizer

# Load the tokenizer and quantized model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer = MarianTokenizer.from_pretrained(model_name)
quantized_model = torch.load("quantized_marianmt.pth")  # Load the full model object

# Set to evaluation mode
quantized_model.eval()

# Run inference
input_text = "Hello, how are you? I am doing great. I want to be able to get to the dining hall on north campus."
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    translated_tokens = quantized_model.generate(**inputs)

for i in range(len(translated_tokens)):
  translated_text = tokenizer.decode(translated_tokens[i], skip_special_tokens=True)
  print("Translated text:", translated_text)

