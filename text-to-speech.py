import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load model & processor (cache them to avoid reloading every time)
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

# Function to transcribe audio
def transcribe_audio(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    inputs = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# Streamlit UI
st.title("🎙️ Speech-to-Text with Wav2Vec2")
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("Transcribing...")
    try:
        text = transcribe_audio(uploaded_file)
        st.success("Transcription:")
        st.write(text)
    except Exception as e:
        st.error(f"Error: {e}")
