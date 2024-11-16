import numpy as np
import librosa
import onnxruntime as ort
import base64

# Load and preprocess the audio file
audio_path = "sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav"
audio, sr = librosa.load(audio_path, sr=16000)

# Generate mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=80
)
mel_spectrogram = np.log(mel_spectrogram + 1e-6)
mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)
print("Mel Spectrogram Shape:", mel_spectrogram.shape)

# Load encoder and decoder ONNX sessions
encoder_session = ort.InferenceSession("sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx")
decoder_session = ort.InferenceSession("sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx")

# Run the encoder model
encoder_input_name = encoder_session.get_inputs()[0].name
encoder_output = encoder_session.run(None, {encoder_input_name: mel_spectrogram})
print("Encoder Output Shape:", encoder_output[0].shape)

# Load token mappings into a dictionary
token_dict = {}
with open("sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            base64_part, token_id_str = parts
            try:
                decoded_token = base64.b64decode(base64_part).decode('utf-8')
            except (UnicodeDecodeError, base64.binascii.Error):
                decoded_token = base64_part
            token_id = int(token_id_str)
            token_dict[token_id] = decoded_token

# Find the start and end token IDs
start_token_id = None
end_token_id = None
for token_id, token in token_dict.items():
    if token == '<|startoftranscript|>':
        start_token_id = token_id
    if token == '<|endoftext|>':
        end_token_id = token_id

print(f"Start Token ID: {start_token_id}")
print(f"End Token ID: {end_token_id}")

if start_token_id is None or end_token_id is None:
    raise ValueError("Start or end token ID not found in token_dict.")

# Initialize decoder inputs
output_text = ""
decoded_tokens = []

tokens = np.array([[start_token_id]], dtype=np.int64)

# Initialize caches
num_decoder_layers = 4  # Adjust based on your model
batch_size = encoder_output[0].shape[1]
max_decode_length = 448
model_dim = 384

in_n_layer_self_k_cache = np.zeros(
    (num_decoder_layers, batch_size, max_decode_length, model_dim),
    dtype=np.float32
)
in_n_layer_self_v_cache = np.zeros_like(in_n_layer_self_k_cache)
n_layer_cross_k = encoder_output[0]
n_layer_cross_v = encoder_output[0]
offset = np.array([0], dtype=np.int64)

# Autoregressive decoding loop
for step in range(100):  # Limit to 100 tokens
    decoder_inputs = {
        "tokens": tokens,
        "in_n_layer_self_k_cache": in_n_layer_self_k_cache,
        "in_n_layer_self_v_cache": in_n_layer_self_v_cache,
        "n_layer_cross_k": n_layer_cross_k,
        "n_layer_cross_v": n_layer_cross_v,
        "offset": offset,
    }
    decoder_output = decoder_session.run(None, decoder_inputs)
    logits = decoder_output[0]

    # Adjust temperature if needed
    temperature = 1.0
    scaled_logits = logits / temperature

    # Use softmax to get probabilities
    e_x = np.exp(scaled_logits - np.max(scaled_logits))
    probabilities = e_x / e_x.sum(axis=-1, keepdims=True)
    predicted_token = np.argmax(probabilities, axis=-1).flatten()[0]

    # Append the token
    decoded_tokens.append(predicted_token)

    # Retrieve the token string
    token_str = token_dict.get(predicted_token, f'<Unknown Token {predicted_token}>')
    print(f"Step {step}: Predicted Token Index: {predicted_token}, Token: {token_str}")

    if predicted_token == end_token_id:
        print("End of sequence token detected. Stopping.")
        break

    output_text += token_str
    tokens = np.array([[predicted_token]], dtype=np.int64)

    # Update caches for next iteration
    in_n_layer_self_k_cache = decoder_output[1]
    in_n_layer_self_v_cache = decoder_output[2]
    offset += 1

print("Transcribed Text:", output_text)

