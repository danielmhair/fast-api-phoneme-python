import os
import csv
from pathlib import Path

import parselmouth
import soundfile as sf
from parselmouth.praat import call

# Set paths
TEXTGRID_DIR = Path("organized_recordings")
AUDIO_DIR = Path("organized_recordings")
OUTPUT_DIR = Path("phoneme_clips")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
MIN_DURATION = 0.08  # seconds
MAX_DURATION = 1.0   # seconds
EXCLUDE_PHONEMES = {"sp", "spn", "sil", ""}

metadata = []

def extract_phonemes(textgrid_path):
    # Read the TextGrid file
    tg = parselmouth.read(str(textgrid_path))
    
    # Find the tier index for the "phones" tier
    num_tiers = call(tg, "Get number of tiers")
    phones_tier_index = None
    for i in range(1, num_tiers + 1):
        tier_name = call(tg, "Get tier name", i)
        if tier_name.lower() == "phones":
            phones_tier_index = i
            break
    if phones_tier_index is None:
        print(f"No phoneme tier found in {textgrid_path.name}")
        return

    audio_path = AUDIO_DIR / textgrid_path.with_suffix(".wav").name
    if not audio_path.exists():
        print(f"Missing audio for {audio_path.name}")
        return

    audio, sr = sf.read(audio_path)
    
    # Get number of intervals in the "phones" tier
    num_intervals = call(tg, "Get number of intervals", phones_tier_index)
    for i in range(1, num_intervals + 1):
        start_time = call(tg, "Get start time of interval", phones_tier_index, i)
        end_time = call(tg, "Get end time of interval", phones_tier_index, i)
        phoneme = call(tg, "Get label of interval", phones_tier_index, i).strip().lower()
        duration = end_time - start_time

        if phoneme in EXCLUDE_PHONEMES:
            continue
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        # Extract audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = audio[start_sample:end_sample]

        # Save segment
        phoneme_dir = OUTPUT_DIR / phoneme
        phoneme_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{textgrid_path.stem}_{i:03d}_{phoneme}.wav"
        filepath = phoneme_dir / filename
        sf.write(filepath, segment, sr)

        metadata.append([filename, phoneme, start_time, end_time])

# Process all TextGrid files
for tg_file in sorted(TEXTGRID_DIR.glob("*.TextGrid")):
    extract_phonemes(tg_file)

# Save metadata CSV
with open(OUTPUT_DIR / "metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "phoneme", "start_time", "end_time"])
    writer.writerows(metadata)

print(f"✅ Done! Extracted {len(metadata)} phoneme clips.")
