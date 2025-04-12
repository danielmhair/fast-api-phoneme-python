import os
import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path

INPUT_BASE = Path("recordings")
OUTPUT_BASE = Path("recordings_trimmed")

MIN_AMP_THRESHOLD = 0.01  # adjust as needed
BUFFER_FRAMES = 1000      # small buffer before/after sound

# Make sure output directory exists
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(INPUT_BASE):
    for file in files:
        if file.endswith(".wav"):
            input_path = Path(root) / file

            # Compute relative path to preserve folder structure
            relative_path = input_path.relative_to(INPUT_BASE)
            output_path = OUTPUT_BASE / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load and resample if needed
            waveform, sample_rate = torchaudio.load(input_path)
            if sample_rate != 16000:
                resample = Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resample(waveform)
                sample_rate = 16000

            waveform = waveform[0]  # Mono

            # Get non-silent region
            nonzero_indices = (waveform.abs() > MIN_AMP_THRESHOLD).nonzero().squeeze()
            if nonzero_indices.numel() == 0:
                print(f"Skipping (silent): {input_path}")
                continue

            if nonzero_indices.ndim == 0:
                # Only one nonzero sample
                idx = nonzero_indices.item()
                start = max(0, idx - BUFFER_FRAMES)
                end = min(len(waveform), idx + BUFFER_FRAMES)
            else:
                start = max(0, nonzero_indices[0].item() - BUFFER_FRAMES)
                end = min(len(waveform), nonzero_indices[-1].item() + BUFFER_FRAMES)

            trimmed_waveform = waveform[start:end]

            # Save trimmed version
            torchaudio.save(output_path.as_posix(), trimmed_waveform.unsqueeze(0), sample_rate)
            print(f"Trimmed saved to: {output_path}")
