import onnxruntime
from transformers import MarianTokenizer, MarianConfig
import numpy as np

# Define the tokenizer and config
model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
config = MarianConfig.from_pretrained(model_name)

# Set up session options
session_options = onnxruntime.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Load ONNX Runtime session
ort_session = onnxruntime.InferenceSession(
    "../models/marianmt_quantized_static.onnx",
    sess_options=session_options,
    providers=['CPUExecutionProvider']
)

# Input text to be translated
input_text = "Where is the furthest point on north campus?"
inputs = tokenizer([input_text], return_tensors="np", max_length=64, truncation=True, padding='max_length')

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Decoder parameters
decoder_start_token_id = config.decoder_start_token_id
eos_token_id = config.eos_token_id
batch_size = input_ids.shape[0]
max_length = 30  # Max length for the output sequence

# Initialize decoder input IDs with start token
decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)
outputs = np.zeros((batch_size, max_length), dtype=np.int64)  # Pre-allocate output array
outputs[:, 0] = decoder_start_token_id  # Fill start token for each batch element

# Decoding loop
for i in range(1, max_length):
    # Prepare inputs for ONNX model
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": outputs[:, :i]  # Use up to the current position
    }
    
    # Run the model
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    
    # Get the next token (argmax over the last dimension)
    next_token_logits = logits[:, -1, :]
    next_token_id = np.argmax(next_token_logits, axis=-1).reshape(-1)
    
    # Append the next token to outputs
    outputs[:, i] = next_token_id
    
    # Check if all sentences have ended
    if np.all(next_token_id == eos_token_id):
        break

# Decode output tokens
translated_tokens = outputs[:, :i+1]  # Slice up to the last generated token
translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
print(translated_text)
print("Translated text:", translated_text[0])

