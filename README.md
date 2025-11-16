
# Audio Projects — SER + ASR (Starter Pack)

This project contains:
- Speech Emotion Recognition (SER) using CNN-LSTM/CRNN
- Speech-to-Text (ASR) using Whisper / Wav2Vec2
- Feature extraction (MFCC, delta, Mel)
- Inference pipeline
- Simulated results screenshot

## How to Run
1. Install:
   pip install librosa tensorflow transformers soundfile matplotlib

2. Train SER:
   python src/train_ser.py

3. ASR (pretrained):
   Use inference.py `transcribe('audio.wav')`

4. SER Prediction:
   predict_emotion(model, 'audio.wav')

## Notes
- Replace template training loops with real dataset loaders.
- Add Whisper/Wav2Vec2 fine-tuning using HuggingFace Trainer.
