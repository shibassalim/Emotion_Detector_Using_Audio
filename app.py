import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib
from pydub import AudioSegment

# Load the trained model
model = joblib.load("emotion_det_model.pkl")

# Feature extraction function
def extract_features(file_name, mfcc=True, chroma=True, mel=True):
    try:
        audio_data, sample_rate = librosa.load(file_name, sr=None)
        features = []

        if mfcc:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            features.extend(np.mean(mfccs.T, axis=0))

        if chroma:
            stft = np.abs(librosa.stft(audio_data))
            chroma_vals = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            features.extend(np.mean(chroma_vals.T, axis=0))

        if mel:
            mel_vals = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            features.extend(np.mean(mel_vals.T, axis=0))

        return np.array(features)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Convert uploaded audio to WAV
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    new_file_path = file_path.replace(".wav", "_converted.wav")
    audio.export(new_file_path, format="wav")
    return new_file_path

# ---------- Streamlit UI ---------- #

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        background-color: #f2f6fc;
        padding: 2rem;
        font-family: "Segoe UI", sans-serif;
    }
    h1 {
        color: #003366;
        text-align: center;
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .emotion {
        text-align: center;
        font-size: 1.5em;
        margin-top: 20px;
        color: #1b4f72;
        font-weight: 600;
    }
    .stApp {
        background-color: #f2f6fc;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Audio Emotion Detection</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, OGG, M4A)", type=['mp3', 'wav', 'ogg', 'm4a'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convert to wav if needed
    if not tmp_path.endswith(".wav"):
        wav_path = convert_to_wav(tmp_path)
    else:
        wav_path = tmp_path

    st.audio(wav_path, format='audio/wav')

    features = extract_features(wav_path)

    if features is not None:
        features = features.reshape(1, -1)
        predicted_emotion = model.predict(features)[0]

        st.markdown(f'<div class="emotion">Predicted Emotion: <span style="color:#0066cc;">{predicted_emotion.capitalize()}</span></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not extract features from the audio file.")
