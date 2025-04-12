import os
import torchaudio

base_dir = "recordings_trimmed"
min_sample_length = 3000

total_files = 0
short_files = 0

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".wav"):
            total_files += 1
            path = os.path.join(root, file)
            try:
                waveform, _ = torchaudio.load(path)
                if waveform.shape[1] < min_sample_length:
                    short_files += 1
            except Exception as e:
                print(f"Failed to load {path}: {e}")

print(f"\nShort files: {short_files}")
print(f"Total files: {total_files}")
print(f"Percentage short: {round(short_files / total_files * 100, 2)}%")
