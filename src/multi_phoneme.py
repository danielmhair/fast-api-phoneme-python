import os
import random
import csv
from pathlib import Path
from pydub import AudioSegment

INPUT_DIR = Path("recordings_t_loud")  # Folder with subfolders named for each phoneme
OUTPUT_DIR = Path("multi_phoneme")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

csv_path = "multi_dataset.csv"
NUM_REPS = 3       # how many times to repeat the same phoneme in one file
SILENCE_MS = 200   # how many ms of silence to insert between repeats
COMBOS_PER_PHONEME = 10  # how many combos to create per phoneme

# 1) Collect all WAV paths, grouped by phoneme
phoneme_to_files = {}

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = Path(root) / file

            # The phoneme is the subfolder name just under INPUT_DIR, i.e. "f", "tʃ", etc.
            # For example, if root == "recordings_t_loud/f", then phoneme = "f"
            rel_path = file_path.relative_to(INPUT_DIR)
            # The first part of rel_path = phoneme directory
            phoneme = rel_path.parts[1]  # e.g. "f", "sh", etc.

            # Accumulate in dict
            if phoneme not in phoneme_to_files:
                phoneme_to_files[phoneme] = []
            phoneme_to_files[phoneme].append(file_path)

# 2) Generate combos and build the CSV
rows = []
rows.append(["path", "phoneme"])  # CSV header

for phoneme, wav_paths in phoneme_to_files.items():
    if not wav_paths:
        continue

    for combo_idx in range(COMBOS_PER_PHONEME):
        # pick random WAV files for each repetition
        chosen_files = random.choices(wav_paths, k=NUM_REPS)
        
        combined = AudioSegment.silent(duration=0)
        for wf in chosen_files:
            seg = AudioSegment.from_wav(wf)
            combined += seg
            combined += AudioSegment.silent(duration=SILENCE_MS)
        
        # label looks like "|f||f||f|" if NUM_REPS=3
        label = "".join(f"|{phoneme}|" for _ in range(NUM_REPS))
        
        # build output path
        out_subdir = OUTPUT_DIR / phoneme
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{phoneme}_rep_{combo_idx}.wav"
        out_path = out_subdir / out_filename
        
        combined.export(out_path, format="wav")
        
        rows.append([str(out_path), label])

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Created multi-phoneme combos in '{OUTPUT_DIR}' and dataset CSV '{csv_path}'")
