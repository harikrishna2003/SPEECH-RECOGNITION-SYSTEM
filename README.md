#  Speech-to-Text Transcription with Wav2Vec2

A Python-based web app that transcribes audio files (`.wav`) into text using Facebook's Wav2Vec2 model. Built with Streamlit for seamless deployment.

## Features
- 🎤 Upload `.wav` audio files via a simple UI
- 🔊 Automatic audio preprocessing (mono conversion, resampling to 16kHz)
- ✍️ Accurate speech-to-text transcription using Wav2Vec2
- 🚀 Fast inference with cached model loading
- ❌ Error handling for invalid inputs

## Technologies Used
- **Python** (Core language)
- **Streamlit** (Web UI)
- **PyTorch & TorchAudio** (Audio processing)
- **Hugging Face Transformers** (Wav2Vec2 model)
- **Wav2Vec2** (`facebook/wav2vec2-base-960h`)

## How It Works
1. **Model Loading**:  
   Pre-trained Wav2Vec2 model and processor are cached for efficiency.
2. **Audio Processing**:  
   - Converts stereo → mono  
   - Resamples to 16kHz (if needed)  
3. **Transcription**:  
   Audio → Wav2Vec2 → Text output  
4. **UI**:  
   Users upload files and receive transcriptions instantly.


## Output

## Installation
```bash
git clone https://github.com/yourusername/speech-to-text-app.git
cd speech-to-text-app
pip install torch torchaudio transformers streamlit 
streamlit run text-to-speech.py  # Launch app
