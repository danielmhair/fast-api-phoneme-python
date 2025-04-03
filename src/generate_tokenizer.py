from pathlib import Path

# Create the vocab content again after kernel reset
phoneme_vocab = ["f", "tʃ", "ʃ", "θ", "|"]
vocab_path = Path("tokenizer") / "phoneme_vocab.txt"
vocab_path.parent.mkdir(parents=True, exist_ok=True)

# Write to file
with open(vocab_path, "w", encoding="utf-8") as f:
    for phoneme in phoneme_vocab:
        f.write(phoneme + "\n")

str(vocab_path)
