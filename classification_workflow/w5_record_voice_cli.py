import sys
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pickle

# Parameters
MIN_AMP_THRESHOLD = 0.01
BUFFER_SAMPLES = 1000   # Buffer (in samples) before/after detected sound
DURATION = 1.0          # Record for 1 second
SAMPLE_RATE = 16000     # Desired sample rate

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    """Record audio for a specified duration."""
    print(f"Recording for {duration} seconds. Please speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio.flatten()

def process_audio(audio):
    """
    Trim silence and normalize the volume.
    Returns the processed audio or None if no significant sound is detected.
    """
    # Find indices where the amplitude exceeds the threshold
    nonzero_indices = np.where(np.abs(audio) > MIN_AMP_THRESHOLD)[0]
    if len(nonzero_indices) == 0:
        return None  # No significant sound detected
    start = max(0, nonzero_indices[0] - BUFFER_SAMPLES)
    end = min(len(audio), nonzero_indices[-1] + BUFFER_SAMPLES)
    trimmed_audio = audio[start:end]
    # Normalize volume so that maximum amplitude is 1
    max_amp = np.max(np.abs(trimmed_audio))
    if max_amp > 0:
        normalized_audio = trimmed_audio / max_amp
    else:
        normalized_audio = trimmed_audio
    return normalized_audio

def extract_embedding(audio, processor, model):
    """Extract a fixed-size embedding from the audio using Wav2Vec2."""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Average the hidden states over time to get a fixed-size vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def main_test_with_voice():
    print("Interactive Phoneme Testing with Processing")
    print("Press Enter to record a sound, or Ctrl+C to exit.")
    
    # Load the Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()
    
    # Load the trained classifier and label encoder
    with open("phoneme_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    while True:
        try:
            input("Press Enter to start recording...")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        
        audio = record_audio()
        processed_audio = process_audio(audio)
        if processed_audio is None:
            print("No significant sound detected. Please try again.")
            continue
        
        print("Processing your recorded sound...")
        embedding = extract_embedding(processed_audio, processor, model)
        embedding = embedding.reshape(1, -1)
        
        # Predict using the classifier
        pred = clf.predict(embedding)
        pred_prob = clf.predict_proba(embedding)[0]
        predicted_phoneme = le.inverse_transform(pred)[0]
        
        print(f"\nPredicted phoneme: {predicted_phoneme}")
        print("Prediction probabilities:")
        for label, prob in zip(le.classes_, pred_prob):
            print(f"  {label}: {prob:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    main_test_with_voice()

