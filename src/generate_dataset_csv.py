import os
import csv

# Path to your recordings root directory
RECORDINGS_ROOT = "recordings_t_loud"
OUTPUT_CSV = "dataset.csv"

# Walk and collect data
data = []

for child_name in os.listdir(RECORDINGS_ROOT):
    child_path = os.path.join(RECORDINGS_ROOT, child_name)
    if not os.path.isdir(child_path):
        continue

    for character in os.listdir(child_path):
        char_path = os.path.join(child_path, character)
        if not os.path.isdir(char_path):
            continue

        for filename in os.listdir(char_path):
            if filename.endswith(".wav"):
                wav_path = os.path.join(child_path, character, filename).replace("\\", "/")
                try:
                    phoneme = filename.split(f"{child_name}_{character}_ep-")[1].split('_')[0].strip()
                    if phoneme:  # skip empty names
                        data.append({"path": wav_path, "phoneme": phoneme})
                except IndexError:
                    print(f"Skipping: {filename} — Unable to parse phoneme.")

# Write to CSV
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["path", "phoneme"])
    writer.writeheader()
    writer.writerows(data)

print(f"Dataset written to {OUTPUT_CSV} with {len(data)} samples.")
