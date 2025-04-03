import os
import torchaudio
from pathlib import Path

INPUT_DIR = Path("recordings_trimmed")
OUTPUT_DIR = Path("recordings_t_loud")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".wav"):
            # Build full input path
            in_path = Path(root) / file

            # Build output path with same relative structure
            rel_path = in_path.relative_to(INPUT_DIR)
            out_path = OUTPUT_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Load audio
            waveform, sample_rate = torchaudio.load(in_path)

            # Normalize amplitude
            max_amp = waveform.abs().max()
            if max_amp > 0:
                waveform = waveform / max_amp

            # Save normalized waveform
            torchaudio.save(str(out_path), waveform, sample_rate)
            print(f"Normalized: {in_path} -> {out_path}")
