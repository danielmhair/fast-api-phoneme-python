import sys
import os
import io
import resampy
import pickle
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor
from starlette.datastructures import UploadFile as StarletteUploadFile

print("Python executable:", sys.executable)

app = FastAPI()

# ✅ Load processor
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base")
# ✅ Load the TorchScript model (which is already traced + mean pooled)
model = torch.jit.load("./wav2vec2_traced_mean.pt")
model.eval()

with open("phoneme_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

MIN_AMP_THRESHOLD = 0.01
BUFFER_SAMPLES = 1000
MIN_REQUIRED_SAMPLES = 1000  # ~0.0625s at 16kHz

def process_audio(audio):
    """Silence detection only."""
    nonzero_indices = np.where(np.abs(audio) > MIN_AMP_THRESHOLD)[0]
    if len(nonzero_indices) == 0:
        return None
    return audio

@app.post("/predict-phoneme-path/{file_path}")
async def predict_phoneme_path(file_path: str):
    base_dir = "../../../Saved/BouncedWavFiles"
    full_path = os.path.abspath(os.path.join(base_dir, os.path.basename(file_path)))
    print(f"Full path: {full_path}")

    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found or is not a file")

    if not full_path.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File must be a WAV file")

    print("Predicting phoneme...")

    with open(full_path, "rb") as f:
        file_bytes = f.read()

    file_like = io.BytesIO(file_bytes)

    upload_file = UploadFile(filename=os.path.basename(full_path), file=file_like)

    result = await predict_phoneme(file=upload_file)
    print(f"Prediction result: {result}")
    return result

@app.post("/predict-phoneme/")
async def predict_phoneme(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        print(f"Loaded {len(audio)} samples at {sr} Hz")
        print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")

        # Convert stereo to mono
        if len(audio.shape) == 2:
            print("Converting stereo to mono...")
            audio = np.mean(audio, axis=1)
            print(f"New shape after mono conversion: {audio.shape}")

        # Silence detection
        if process_audio(audio) is None:
            return JSONResponse(status_code=400, content={"error": "No significant sound detected"})

        # Pad short audio
        if len(audio) < MIN_REQUIRED_SAMPLES:
            print(f"Padding short audio from {len(audio)} to {MIN_REQUIRED_SAMPLES} samples")
            audio = np.pad(audio, (0, MIN_REQUIRED_SAMPLES - len(audio)), mode='constant')

        # Resample if needed
        if sr != 16000:
            print(f"Resampling from {sr} to 16000 Hz")
            audio = resampy.resample(audio, sr, 16000)
            sr = 16000
            print(f"Resampled length: {len(audio)}")

        # Normalize
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        # Generate embedding
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=False) # type: ignore[call-overload]
        with torch.no_grad():
            embedding = model(inputs["input_values"]).squeeze().numpy().reshape(1, -1)

        pred = clf.predict(embedding)
        pred_prob = clf.predict_proba(embedding)[0]
        predicted_phoneme = le.inverse_transform(pred)[0]

        probs = {label: float(prob) for label, prob in zip(le.classes_, pred_prob)}

        return {"phoneme": predicted_phoneme, "probabilities": probs}

    except Exception as e:
        print(f"Exception: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
