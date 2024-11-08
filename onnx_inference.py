import onnxruntime as ort
from transformers import MarianTokenizer
import numpy as np

# Load tokenizer and ONNX model
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer = MarianTokenizer.from_pretrained(model_name)
onnx_model_path = "marianmt_quantized.onnx"

# Create an ONNX Runtime session
session = ort.InferenceSession(onnx_model_path)

# Input text and tokenization
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="np")

# Define the valid start token (decoder_input_ids)
decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id < tokenizer.vocab_size else tokenizer.bos_token_id
if decoder_start_token_id is None or decoder_start_token_id >= tokenizer.vocab_size:
    raise ValueError("The start token ID is out of vocabulary range. Please check tokenizer configuration.")

decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

# Use "output" as the output name
output_name = "output"

# Run inference
try:
    output = session.run(
        output_names=[output_name],
        input_feed={
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "decoder_input_ids": decoder_input_ids
        }
    )
    
    # Decode and print the output
    translated_tokens = output[0]
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print("Translated text:", translated_text)

except Exception as e:
    print("Error during ONNX inference:", e)

