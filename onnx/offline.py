import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer

# Paths to the saved tokenizer and ONNX model
tokenizer_dir = "../models/tokenizer"
onnx_model_path = "../models/marianmt_quantized_static_tokenize.onnx"

# 1. Load the tokenizer
tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)

# 2. Load the ONNX model
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# 3. Preprocess the input text
input_text = "where is the nearest hotel"
inputs = tokenizer([input_text], return_tensors="np", max_length=64, truncation=True, padding="max_length")

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Prepare decoder input IDs with the start token
decoder_start_token_id = tokenizer.pad_token_id  # Typically pad_token_id is used as the start token
decoder_input_ids = np.full((input_ids.shape[0], 1), decoder_start_token_id, dtype=np.int64)

# 4. Perform inference in a loop for auto-regressive generation
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
    logits = logits[:, -1, :]  # Focus on the logits of the last generated token

    # Apply repetition penalty
    repetition_penalty = 1.2
    for batch_idx, seq in enumerate(outputs):
        for token in seq:
            logits[batch_idx, token] /= repetition_penalty

    # Apply length penalty
    length_penalty = 10.0  # Adjust this value (e.g., 0.6 favors longer sequences, >1 favors shorter ones)
    for batch_idx in range(logits.shape[0]):
        logits[batch_idx] /= (i + 1) ** length_penalty

    # Get the next token ID (greedy decoding: argmax)
    next_token_id = np.argmax(logits, axis=-1).reshape(-1)
    outputs[:, i] = next_token_id  # Append the token to the outputs

    # Stop if all sentences have reached the EOS token
    if np.all(next_token_id == tokenizer.eos_token_id):
        break

# 5. Postprocess the outputs
decoded_outputs = tokenizer.batch_decode(outputs[:, :i+1], skip_special_tokens=True)


# Display the result
print("Input:", input_text)
print("Translation:", decoded_outputs[0])
