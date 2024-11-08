import onnxruntime
from transformers import MarianTokenizer, MarianConfig
import numpy as np

# Replace with your desired model
model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
config = MarianConfig.from_pretrained(model_name)

ort_session = onnxruntime.InferenceSession("marianmt_quantized.onnx")
input_text = "Hello world!"
inputs = tokenizer([input_text], return_tensors="np")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']


decoder_start_token_id = config.decoder_start_token_id
eos_token_id = config.eos_token_id

batch_size = input_ids.shape[0]
decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)


# Start decoding
outputs = decoder_input_ids
max_length = 50  # Set a maximum output length

for _ in range(max_length):
    # Prepare inputs for the model
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": outputs
    }
    # Run the model
    ort_outs = ort_session.run(None, ort_inputs)
    # Get the logits
    logits = ort_outs[0]
    # Get the next token logits (the last time step)
    next_token_logits = logits[:, -1, :]
    # Get the most probable next token
    next_token_id = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)
    # Append the next token id to outputs
    outputs = np.concatenate([outputs, next_token_id], axis=1)
    # Check for end of sentence token
    if np.all(next_token_id == eos_token_id):
        break

# Decode the output ids
translated_tokens = outputs
translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print("Translated text:", translated_text[0])


