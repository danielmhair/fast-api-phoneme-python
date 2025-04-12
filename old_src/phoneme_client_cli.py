import sounddevice as sd
import numpy as np
import requests
import soundfile as sf
import io

# Parameters

DURATION = 1.0
SAMPLE_RATE = 16000
API_URL = "http://localhost:8000/predict-phoneme-path/"

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"Recording for {duration} seconds. Please speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio

def send_audio(audio):
    response = requests.post(API_URL + audio)

    if response.status_code == 200:
        data = response.json()
        print(f"\nPredicted phoneme: {data['phoneme']}")
        print("Prediction probabilities:")
        for label, prob in data["probabilities"].items():
            print(f"  {label}: {prob:.2f}")
        print("-" * 40)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def main():
    print("Interactive Phoneme Testing via API")
    print("Press Enter to record a sound, or Ctrl+C to exit.")

    while True:
        try:
            input("Press Enter to start recording...")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        # print("Recording audio...")
        # audio = record_audio()
        print("Sending audio to API...")
        send_audio("7878.wav")

if __name__ == "__main__":
    main()
