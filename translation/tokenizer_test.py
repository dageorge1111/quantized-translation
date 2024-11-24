import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer

# Paths to the saved tokenizer and quantized ONNX model
tokenizer_dir = "../models/tokenizer"
quantized_model_path = "../models/marianmt_model_int8.onnx"

# Load the tokenizer
tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)

# Load the INT16 quantized ONNX model
ort_session = ort.InferenceSession(quantized_model_path, providers=["CPUExecutionProvider"])

# Preprocess the input text
input_text = "Why do you like Germany so much?"
inputs = tokenizer([input_text], return_tensors="np", max_length=64, truncation=True, padding="max_length")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# Print the input tokens
print("Input text:", input_text)
print("Input token IDs:")
print(input_ids)
print("Decoded input tokens:")
print(tokenizer.batch_decode(input_ids, skip_special_tokens=False))
