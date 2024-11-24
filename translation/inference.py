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
# Prepare decoder input IDs with the start token
decoder_start_token_id = tokenizer.pad_token_id  # Typically `pad_token_id` is used as the start token
decoder_input_ids = np.full((input_ids.shape[0], 1), decoder_start_token_id, dtype=np.int64)

# Perform inference in a loop for auto-regressive generation
max_length = 30  # Define the maximum output sequence length
outputs = np.zeros((input_ids.shape[0], max_length), dtype=np.int64)
outputs[:, 0] = decoder_start_token_id  # Initialize with the start token

for i in range(1, max_length):
    # Prepare ONNX inputs
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": outputs[:, :i]
    }

    # Run the ONNX model
    ort_outs = ort_session.run(None, ort_inputs)

    # Extract logits for the last position
    logits = ort_outs[0]  # Shape: (batch_size, sequence_length, vocab_size)
    next_token_logits = logits[:, -1, :]  # Extract logits for the last generated token

    # Get the next token ID (greedy decoding: argmax)
    next_token_id = np.argmax(next_token_logits, axis=-1).reshape(-1)
    outputs[:, i] = next_token_id  # Append the token to the outputs

    # Stop if all sentences have reached the EOS token
    if np.all(next_token_id == tokenizer.eos_token_id):
        break
print("Final token IDs (before decoding):")
print(outputs)

# Decode the output sequences
# Print the tokenizer's configuration
print("Tokenizer configuration:", tokenizer.init_kwargs)

# If available, print the vocab file path
if "vocab_file" in tokenizer.init_kwargs:
    print("SentencePiece model path:", tokenizer.init_kwargs["vocab_file"])
else:
    print("SentencePiece model path not found in init_kwargs.")

decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Display the translation result
print("Input:", input_text)
print("Translation:", decoded_outputs[0])

