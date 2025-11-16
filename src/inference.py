
# src/inference.py
import librosa
import numpy as np
from transformers import pipeline
from src.features import extract_features

# Emotion model placeholder
def predict_emotion(model, audio_path):
    feat = extract_features(audio_path)
    feat = np.expand_dims(feat, axis=0)
    pred = model.predict(feat)[0]
    emotion = int(np.argmax(pred))
    return emotion

# Whisper/Wav2Vec2 ASR
def transcribe(audio_path):
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return asr(audio_path)["text"]
