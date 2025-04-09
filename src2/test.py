import json
import pandas as pd

vocab = json.load(open("vocab.json"))
df = pd.read_csv("dataset.csv")

missing = set()
for label in df["phoneme"]:
    if label not in vocab:
        missing.add(label)

print("Missing phonemes:", missing)