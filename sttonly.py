import torch
import pyaudio
import numpy as np
import streamlit as st
from transformers import pipeline
import wave

# ---- Hugging Face API Token ----
HF_AUTH_TOKEN = "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"  # Fetch from environment variables

if not HF_AUTH_TOKEN:
    st.error("❌ Hugging Face API Token not found! Set it in your environment variables.")
    st.stop()

# Load Whisper model for speech-to-text
@st.cache_resource
def load_whisper_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

pipe = load_whisper_model()

# Initialize session state
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

# Streamlit UI
st.title("🎙️ Speech-to-Text Transcription")
st.write("Record your speech, and we'll transcribe it for you.")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  
CHUNK = 1024
RECORD_SECONDS = 30  # Shortened for quick testing
AUDIO_FILE = "recorded_audio.wav"

# Start Recording
if st.button("🎤 Start Recording"):
    st.write("🎙️ Recording for 30 seconds... Please speak.")

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):  
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    # Stop and close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as WAV file
    wf = wave.open(AUDIO_FILE, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    st.success("✅ Recording complete!")

    # Convert to NumPy array & normalize
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0   
    st.session_state.audio_data = audio_data  # Save audio in session state

    # Transcribe speech
    st.write("🛠️ Transcribing speech...")
    transcription = pipe({"array": audio_data, "sampling_rate": RATE})["text"]
    st.session_state.transcription = transcription  # Save transcription in session state
    st.success("✅ Transcription Complete!")

# Show transcription if available
if st.session_state.transcription:
    st.subheader("🗣️ Transcription:")
    st.write(st.session_state.transcription)