import os
import csv
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Set paths
PHONEME_CLIPS_DIR = Path("organized_recordings")
EMBEDDINGS_OUTPUT_DIR = Path("phoneme_embeddings")

def extract_embeddings_for_phonemes():
    if not PHONEME_CLIPS_DIR.exists():
        print(f"Error: {PHONEME_CLIPS_DIR} does not exist. Be sure to run w1_prepare_wav_files, but should run with 0_workflow.py")
        return
    # Clean previous embeddings
    if EMBEDDINGS_OUTPUT_DIR.exists():
        for item in EMBEDDINGS_OUTPUT_DIR.iterdir():
            if item.is_file():
                item.unlink()
            else:
                os.rmdir(item)
        EMBEDDINGS_OUTPUT_DIR.rmdir()
    EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory {EMBEDDINGS_OUTPUT_DIR} for saving embeddings.")

    # Load Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()

    metadata = []

    # Debug: print number of phoneme subdirectories
    phoneme_dirs = list(PHONEME_CLIPS_DIR.iterdir())
    print(f"Found {len(phoneme_dirs)} subdirectories in {PHONEME_CLIPS_DIR}")

    # Loop through each phoneme directory
    for phoneme_dir in sorted(PHONEME_CLIPS_DIR.iterdir()):
        if not phoneme_dir.is_dir():
            continue
        # List all WAV files in this directory
        wav_files = list(phoneme_dir.glob("*.wav"))
        print(f"In directory '{phoneme_dir.name}', found {len(wav_files)} WAV files.")
        
        for wav_file in sorted(wav_files):
            try:
                # Load audio
                audio, sr = sf.read(str(wav_file))
            except Exception as e:
                print(f"Error reading {wav_file}: {e}")
                continue

            if sr != 16000:
                print(f"Warning: {wav_file.name} has sample rate {sr}, expected 16000.")
            
            # Process audio with the processor
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True) # type: ignore
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Average the hidden states over time to get a fixed-size embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Save the embedding as a .npy file
            embedding_filename = wav_file.stem + ".npy"
            np.save(EMBEDDINGS_OUTPUT_DIR / embedding_filename, embedding)
            
            # Record metadata
            metadata.append([wav_file.name, phoneme_dir.name, embedding_filename])

    # Save metadata CSV
    metadata_csv = EMBEDDINGS_OUTPUT_DIR / "metadata.csv"
    with open(metadata_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_filename", "phoneme", "embedding_filename"])
        writer.writerows(metadata)

    print(f"✅ Done! Extracted embeddings for {len(metadata)} phoneme clips.")


if __name__ == "__main__":
    extract_embeddings_for_phonemes()