import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
import csv
from pathlib import Path

EMBEDDINGS_DIR = Path("phoneme_embeddings")
METADATA_PATH = EMBEDDINGS_DIR / "metadata.csv"


def load_embeddings_and_labels():
    embeddings = []
    labels = []

    with open(METADATA_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb_file = EMBEDDINGS_DIR / row["embedding_filename"]
            if emb_file.exists():
                emb = np.load(emb_file)
                embeddings.append(emb)
                labels.append(row["phoneme"])

    return np.array(embeddings), np.array(labels)


def analyze_confusion():
    # Load models
    with open("dist/phoneme_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("dist/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    X, labels = load_embeddings_and_labels()
    y = le.transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    class_labels = le.inverse_transform(np.arange(len(le.classes_)))

    # Find most confused pairs (excluding diagonal)
    confusion_pairs = []
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((class_labels[i], class_labels[j], cm[i, j]))

    confusion_pairs.sort(key=lambda x: -x[2])

    print("🔎 Top 10 Most Confused Phoneme Pairs:")
    for true_label, pred_label, count in confusion_pairs[:10]:
        print(f"  - {true_label} mistaken as {pred_label}: {count} times")

    print("⚠️ Phonemes with Lowest Precision (under 0.75):")
    for label in class_labels:
        if label in report and isinstance(report[label], dict):
            precision = report[label].get("precision", 1.0) # type:ignore
            if precision < 0.75:
                print(f"  - {label}: precision = {precision:.2f}")

if __name__ == "__main__":
    analyze_confusion()