import numpy as np
import librosa
import onnxruntime as ort
import base64

# Load and preprocess the audio file
audio_path = "sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav"
audio, sr = librosa.load(audio_path, sr=16000)

# Generate mel spectrogram with 80 mel channels
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=80)
mel_spectrogram = np.log(mel_spectrogram + 1e-6)
mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)
print("Mel Spectrogram Shape:", mel_spectrogram.shape)

# Load encoder and decoder ONNX sessions
encoder_session = ort.InferenceSession("sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx")
decoder_session = ort.InferenceSession("sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx")

# Run the encoder model with the mel spectrogram
encoder_input_name = encoder_session.get_inputs()[0].name
encoder_output = encoder_session.run(None, {encoder_input_name: mel_spectrogram})
print("Encoder Output Shape:", encoder_output[0].shape)
print("Encoder Output Detailed:", encoder_output)  # Check encoder output contents

# Load token mappings from tiny.en-tokens.txt
token_list = []
with open("sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if parts and len(parts) > 1:
            base64_part = parts[0]
            try:
                decoded_token = base64.b64decode(base64_part).decode('utf-8')
                token_list.append(decoded_token)
            except (UnicodeDecodeError, base64.binascii.Error):
                token_list.append(base64_part)

print("Decoded Token List Sample:", token_list[:10])

# Initialize decoder inputs
output_text = ""
tokens = np.array([[1]], dtype=np.int64)  # Try different start token (e.g., 1)
in_n_layer_self_k_cache = np.zeros((4, encoder_output[0].shape[0], 448, 384), dtype=np.float32)
in_n_layer_self_v_cache = np.zeros((4, encoder_output[0].shape[0], 448, 384), dtype=np.float32)
n_layer_cross_k = encoder_output[0]
n_layer_cross_v = encoder_output[0]
offset = np.array([0], dtype=np.int64)

# Autoregressive decoding loop with temperature scaling
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

    # Adjust temperature to encourage diversity
    temperature = 1.0
    scaled_logits = logits / temperature
    predicted_token = np.argmax(scaled_logits, axis=-1).flatten()[0]

    # Debugging: Print each step and predicted token
    print(f"Step {step}: Predicted Token Index: {predicted_token}, Token: {token_list[predicted_token] if predicted_token < len(token_list) else 'N/A'}")

    if predicted_token == 50256:
        print("End of sequence token detected. Stopping.")
        break

    output_text += token_list[predicted_token]
    tokens = np.array([[predicted_token]], dtype=np.int64)

    # Update caches for next iteration
    in_n_layer_self_k_cache = decoder_output[1]
    in_n_layer_self_v_cache = decoder_output[2]
    offset += 1

print("Transcribed Text:", output_text)
# Initialize a list to store the first 15 tokens
first_15_tokens = []

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

    # Temperature scaling for randomness (optional)
    temperature = 1.0
    scaled_logits = logits / temperature
    predicted_token = np.argmax(scaled_logits, axis=-1).flatten()[0]

    # Append to first_15_tokens list
    if len(first_15_tokens) < 15:
        first_15_tokens.append(token_list[predicted_token] if predicted_token < len(token_list) else 'N/A')
    print(first_15_tokens)
    # Break if EOS token (e.g., 50256) is predicted
    if predicted_token == 502560:
        print("End of sequence token detected. Stopping.")
        break

    # Append predicted token to output_text for the full transcription
    output_text += token_list[predicted_token]
    tokens = np.array([[predicted_token]], dtype=np.int64)  # Update tokens input for next step

    # Update caches for next iteration
    in_n_layer_self_k_cache = decoder_output[1]
    in_n_layer_self_v_cache = decoder_output[2]
    offset += 1

# Print the first 15 tokens
print("First 15 Tokens:", first_15_tokens)
print("Transcribed Text:", output_text)

