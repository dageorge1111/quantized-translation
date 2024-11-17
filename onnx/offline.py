import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer

# Paths to the saved tokenizer and ONNX model
tokenizer_dir = "../models/tokenizer"
onnx_model_path = "../models/marianmt_model_float32.onnx"

# 1. Load the tokenizer
tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)

# 2. Load the ONNX model
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# 3. Preprocess the input text
input_text = "Where is the nearest hotel?"
inputs = tokenizer([input_text], return_tensors="np", max_length=64, truncation=True, padding="max_length")

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 4. Initialize dynamic outputs
decoder_start_token_id = tokenizer.pad_token_id
outputs = [[decoder_start_token_id] for _ in range(input_ids.shape[0])]

# 5. Perform auto-regressive decoding
max_length = 30
for i in range(1, max_length):
    # Prepare ONNX inputs
    decoder_input_ids = np.array([seq for seq in outputs], dtype=np.int64)
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids
    }

    # Run the ONNX model
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0][:, -1, :]  # Logits for the last generated token

    # Get the next token IDs
    next_token_id = np.argmax(logits, axis=-1)

    # Append next tokens to outputs
    for batch_idx, token_id in enumerate(next_token_id):
        outputs[batch_idx].append(token_id)

    # Stop decoding if all sequences have reached the EOS token
    if np.all(next_token_id == tokenizer.eos_token_id):
        break

# 6. Decode outputs to text
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Input:", input_text)
print("Translation:", decoded_outputs[0])

