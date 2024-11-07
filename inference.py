import torch
from transformers import MarianMTModel, MarianTokenizer

# Step 1: Load the quantized model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Replace with your specific MarianMT model
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Use the previously defined function to load the quantized model
quantized_model = load_quantized_model(model_name, "quantized_marianmt.pth")

# Set the model to evaluation mode
quantized_model.eval()

# Step 2: Define your input text for translation
input_text = "Hello, how are you?"  # Replace with the text you want to translate
inputs = tokenizer(input_text, return_tensors="pt")

# Step 3: Perform inference
with torch.no_grad():
    translated_tokens = quantized_model.generate(**inputs)

# Step 4: Decode the translated tokens
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print("Translated text:", translated_text)

