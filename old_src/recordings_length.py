import os
import torchaudio

# Change this to your trimmed directory
INPUT_BASE = "recordings_trimmed"

min_length = float("inf")
shortest_file = None

sample_rate = None

for root, dirs, files in os.walk(INPUT_BASE):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                length = waveform.shape[1]
                if length < min_length:
                    min_length = length
                    shortest_file = file_path
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

print(f"Shortest file: {shortest_file}")
print(f"Length (samples): {min_length}")
if sample_rate is not None:
    print(f"Length (seconds): {min_length / sample_rate:.4f} sec")
