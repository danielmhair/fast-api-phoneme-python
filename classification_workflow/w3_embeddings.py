import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Set paths
EMBEDDINGS_DIR = Path("phoneme_embeddings")
metadata_path = EMBEDDINGS_DIR / "metadata.csv"

def embeddings():
    # Load metadata and embeddings
    embeddings = []
    labels = []

    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emb_file = EMBEDDINGS_DIR / row["embedding_filename"]
            if emb_file.exists():
                emb = np.load(emb_file)
                embeddings.append(emb)
                labels.append(row["phoneme"])
            else:
                print(f"Missing embedding file: {row['embedding_filename']}")

    embeddings = np.array(embeddings)
    print("Embeddings shape:", embeddings.shape)

    # Check class distribution
    class_counts = Counter(labels)
    print("Class distribution before filtering:", class_counts)

    # Filter out classes with fewer than 2 samples
    filtered_data = [(emb, label) for emb, label in zip(embeddings, labels) if class_counts[label] >= 2]
    if not filtered_data:
        raise ValueError("No classes have at least 2 samples!")
    filtered_embeddings, filtered_labels = zip(*filtered_data)
    filtered_embeddings = np.array(filtered_embeddings)
    print("Filtered embeddings shape:", filtered_embeddings.shape)
    print("Filtered class distribution:", Counter(filtered_labels))

    # Encode the phoneme labels into integers
    le = LabelEncoder()
    y = le.fit_transform(filtered_labels)

    # Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        filtered_embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and train an MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Use only the classes present in the test set
    unique_test_labels = np.unique(y_test)
    target_names = [le.inverse_transform([label])[0] for label in unique_test_labels]

    report = classification_report(y_test, y_pred, labels=unique_test_labels, target_names=target_names)

    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    print(report)

    # Save the trained classifier and label encoder
    import pickle

    with open("phoneme_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Classifier and label encoder saved.")
    print("✅ Embedding extraction and classification complete!")

if __name__ == "__main__":
    embeddings()