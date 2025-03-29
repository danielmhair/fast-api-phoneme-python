import os
import csv
import requests
import Levenshtein
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000/predict-phonemes"
AUDIO_ROOT = "recordings"
MAX_EDIT_DISTANCE = 0
CSV_OUTPUT = "evaluation_results.csv"
CONFUSION_MATRIX_IMAGE = "confusion_matrix.png"

phoneme_results = defaultdict(lambda: {"total": 0, "correct": 0, "mistakes": []})
confusion_matrix_data = defaultdict(lambda: defaultdict(int))

def get_prediction(file_path):
    with open(file_path, "rb") as audio_file:
        response = requests.post(API_URL, files={"audio": audio_file})
        if response.status_code == 200:
            return response.json().get("phonemes", "").strip()
        return None

def evaluate():
    total = 0
    correct = 0
    csv_rows = []

    for phoneme in os.listdir(AUDIO_ROOT):
        phoneme_dir = os.path.join(AUDIO_ROOT, phoneme)
        if not os.path.isdir(phoneme_dir):
            continue

        for child in os.listdir(phoneme_dir):
            child_dir = os.path.join(phoneme_dir, child)
            if not os.path.isdir(child_dir):
                continue

            for filename in os.listdir(child_dir):
                if not filename.endswith(".wav"):
                    continue

                file_path = os.path.join(child_dir, filename)
                prediction = get_prediction(file_path)

                phoneme_results[phoneme]["total"] += 1
                total += 1

                row = {
                    "expected": phoneme,
                    "predicted": prediction or "empty",
                    "filename": filename,
                    "correct": False
                }

                confusion_matrix_data[phoneme][prediction or "empty"] += 1

                if prediction and Levenshtein.distance(prediction, phoneme) <= MAX_EDIT_DISTANCE:
                    phoneme_results[phoneme]["correct"] += 1
                    correct += 1
                    row["correct"] = True
                else:
                    phoneme_results[phoneme]["mistakes"].append((filename, prediction))

                csv_rows.append(row)

    # Write CSV
    with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "expected", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print accuracy
    for phoneme, stats in phoneme_results.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"Phoneme '{phoneme}': {stats['correct']}/{stats['total']} correct ({acc:.2f}%)")

    overall = correct / total * 100 if total > 0 else 0
    print(f"\n✅ Overall accuracy: {correct}/{total} ({overall:.2f}%)")

    # Create and save confusion matrix
    df = pd.DataFrame(confusion_matrix_data).fillna(0).astype(int).T
    plt.figure(figsize=(16, 12))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Phoneme Confusion Matrix")
    plt.ylabel("Expected")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMAGE)
    print(f"\n📄 CSV saved to {CSV_OUTPUT}")
    print(f"🖼️  Confusion matrix image saved to {CONFUSION_MATRIX_IMAGE}")

evaluate()
