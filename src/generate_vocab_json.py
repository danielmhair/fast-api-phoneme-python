import csv
import json

CSV_PATH = "dataset.csv"
VOCAB_OUTPUT = "vocab.json"

phoneme_set = set()

with open(CSV_PATH, mode="r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        phoneme = row["phoneme"].strip()
        phoneme_set.add(phoneme)  # No list() — keep the full string!

# Sort to keep it consistent
phoneme_list = sorted(phoneme_set)

# Map each phoneme to a unique ID
vocab = {symbol: idx for idx, symbol in enumerate(phoneme_list)}
vocab["|"] = len(vocab)  # Add blank token (for CTC)
vocab["[UNK]"] = len(vocab)  # Optional: unknown token

# Save to file
with open(VOCAB_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Vocabulary saved to {VOCAB_OUTPUT} with {len(vocab)} tokens.")
