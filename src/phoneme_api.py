import sys
import resampy
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import torch
import pickle
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from starlette.datastructures import UploadFile as StarletteUploadFile
import io
import soundfile as sf
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

def process_audio(audio):
    nonzero_indices = np.where(np.abs(audio) > MIN_AMP_THRESHOLD)[0]
    if len(nonzero_indices) == 0:
        return None
    start = max(0, nonzero_indices[0] - BUFFER_SAMPLES)
    end = min(len(audio), nonzero_indices[-1] + BUFFER_SAMPLES)
    trimmed_audio = audio[start:end]
    max_amp = np.max(np.abs(trimmed_audio))
    if max_amp > 0:
        normalized_audio = trimmed_audio / max_amp
    else:
        normalized_audio = trimmed_audio
    return normalized_audio

@app.post("/predict-phoneme-path/{file_path}")
async def predict_phoneme_path(file_path: str):
    base_dir = "../../../Saved/BouncedWavFiles"
    # Currently: Full path: ../../Saved/BouncedWavFiles\5023.wav - I want it a full path
    full_path = os.path.join(os.path.curdir, base_dir, os.path.basename(file_path))
    print(f"Full path: {full_path}")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    print(f"File exists: {full_path}")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File does not exist")
    print(f"Is a file: {full_path}")
    if not full_path.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File must be a WAV file")
    print("Predicting phoneme...")
    with open(full_path, "rb") as f:
        file_bytes = f.read()

    file_like = io.BytesIO(file_bytes)

    # ✅ Use FastAPI's UploadFile class for compatibility with type checker
    upload_file = UploadFile(filename=os.path.basename(full_path), file=file_like)  # type: ignore

    result = await predict_phoneme(file=upload_file)
    print(f"Prediction result: {result}")
    return result

@app.post("/predict-phoneme/")
async def predict_phoneme(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        print(f"Loaded {len(audio)} samples at {sr} Hz")
        print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}, sr: {sr}")

        if sr != 16000:
            audio = resampy.resample(audio, sr, 16000)
            sr = 16000

        # Ensure 1D shape for mono audio
        if len(audio.shape) == 2 and audio.shape[1] == 1:
            audio = audio.flatten()
        
        if len(audio.shape) == 2:
            # Average across channels to ensures compatibility with Wav2Vec2, which expects 1D mono waveforms
            audio = np.mean(audio, axis=1)

        if process_audio(audio) is None:
            return JSONResponse(status_code=400, content={"error": "No significant sound detected"})
        
        # Pad short audio
        if len(audio) < 1000:
            audio = np.pad(audio, (0, 1000 - len(audio)), mode='constant')

        # Normalize audio
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
        with torch.no_grad():
            embedding = model(inputs["input_values"]).squeeze().numpy().reshape(1, -1)

        pred = clf.predict(embedding)
        pred_prob = clf.predict_proba(embedding)[0]
        predicted_phoneme = le.inverse_transform(pred)[0]

        probs = {label: float(prob) for label, prob in zip(le.classes_, pred_prob)}

        return {"phoneme": predicted_phoneme, "probabilities": probs}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
