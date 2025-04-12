import sys
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pickle

def load_audio(file_path, target_sr=16000):
    audio, sr = sf.read(file_path)
    if sr != target_sr:
        raise ValueError(f"Sample rate of {file_path} is {sr}, expected {target_sr}.")
    return audio

def extract_embedding(audio, processor, model):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Average over time dimension to get fixed-size vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def main():        
    folder_path = Path("organized_recordings")
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist.")
        sys.exit(1)
        
    # Load the Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    
    # Load the trained classifier and label encoder
    with open("phoneme_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    # Iterate over all WAV files in the folder (recursively)
    wav_files = list(folder_path.glob("**/*.wav"))
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        sys.exit(0)
    
    print(f"Found {len(wav_files)} WAV files. Processing...")
    
    for wav_file in wav_files:
        try:
            audio = load_audio(str(wav_file))
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue
        
        # Extract embedding from the audio file
        embedding = extract_embedding(audio, processor, model)
        embedding = embedding.reshape(1, -1)
        
        # Predict phoneme using classifier
        pred = clf.predict(embedding)
        pred_prob = clf.predict_proba(embedding)[0]
        predicted_phoneme = le.inverse_transform(pred)[0]
        
        # Print the result for the file
        print(f"File: {wav_file}")
        print(f"  Predicted phoneme: {predicted_phoneme}")
        print("  Prediction probabilities:")
        for label, prob in zip(le.classes_, pred_prob):
            print(f"    {label}: {prob:.2f}")
        print("-" * 40)
    
if __name__ == "__main__":
    main()
