import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import wavio

# Load Whisper model
model = whisper.load_model("tiny")

# Define recording parameters
sample_rate = 16000  # Whisper works best with 16 kHz sample rate
duration = 5  # Duration of recording in seconds

print("Recording...")

# Record audio
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
print("Recording complete.")

# Save the recorded audio to a temporary WAV file
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
    wavio.write(tmpfile.name, audio, sample_rate, sampwidth=2)
    audio_path = tmpfile.name

# Transcribe the audio file with Whisper
result = model.transcribe(audio_path)
print("Transcription:", result["text"])

# Clean up temporary file
os.remove(audio_path)

