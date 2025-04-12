import os
import shutil
import csv
from pathlib import Path

# Set source and target directories
SOURCE_DIR = Path("recordings_t_loud")
TARGET_DIR = Path("organized_recordings")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Prepare metadata list
metadata = []

# Iterate over child folders in the source directory
for child_dir in SOURCE_DIR.iterdir():
    if child_dir.is_dir():
        child_name = child_dir.name
        # Iterate over phoneme (letter) folders within each child folder
        for phoneme_dir in child_dir.iterdir():
            if phoneme_dir.is_dir():
                phoneme = phoneme_dir.name.lower()  # the letter (phoneme)
                # Create target subdirectory for this phoneme if it doesn't exist
                target_subdir = TARGET_DIR / phoneme
                target_subdir.mkdir(parents=True, exist_ok=True)
                
                # Counter for filenames per phoneme per child
                counter = 1
                # Iterate over all WAV files in the current phoneme folder
                for wav_file in phoneme_dir.glob("*.wav"):
                    # Create a new filename: childname_phoneme_counter.wav
                    new_filename = f"{child_name}_{phoneme}_{counter:03d}.wav"
                    target_path = target_subdir / new_filename
                    
                    # Copy the file to the new location
                    shutil.copy2(wav_file, target_path)
                    
                    # Record metadata: new filename, child name, phoneme, original path
                    metadata.append([new_filename, child_name, phoneme, str(wav_file)])
                    
                    counter += 1

# Save metadata CSV in the target directory
metadata_csv_path = TARGET_DIR / "metadata.csv"
with open(metadata_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["new_filename", "child_name", "phoneme", "original_file"])
    writer.writerows(metadata)

print(f"✅ Reorganization complete! Files have been organized into '{TARGET_DIR}'.")
print(f"Metadata CSV saved at: {metadata_csv_path}")
