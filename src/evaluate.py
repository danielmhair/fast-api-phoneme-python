import os
import requests
from collections import defaultdict

from torch import flatten
import json
# Config
model_url = "http://localhost:8000/predict-phonemes"
dataset_dir = "recordings_t_loud"  # Top-level directory: recordings/child_name/character/*.wav

# Character to expected phoneme map (example subset — expand as needed)
char_to_phoneme = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f",
    "g": "g", "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l",
    "m": "m", "n": "n", "o": "ɒ", "p": "p", "q": "k", "r": "r",
    "s": "s", "t": "t", "u": "ʌ", "v": "v", "w": "w", "x": "ks",
    "y": "j", "z": "z", "ch": "tʃ", "sh": "ʃ", "th": "θ", "dh": "ð", "ng": "ŋ"
}

# Phoneme similarity groups
phoneme_similarity_groups = [
    {"tʃ", "dʒ"}, {"θ", "ð"}, {"s", "z"}, {"ʃ", "ʒ"},
    {"b", "p"}, {"d", "t"}, {"g", "k"}, {"v", "f"},
    {"n", "ŋ"}, {"æ", "a", "ɑ", "ʌ"}, {"e", "ɛ"}, {"i", "ɪ"}, {"u", "ʊ"},
    {"r", "ɹ"}, {"l", "ɫ"}
]

# Build similarity lookup
phoneme_to_group = {}
for group in phoneme_similarity_groups:
    for phoneme in group:
        phoneme_to_group[phoneme] = group

def are_similar(p1: str, p2: str) -> bool:
    if (not p1 and p2) or (not p2 and p1):
        return False
    if p1 == p2:
        return True
    # if 'tʃi' then 'tʃ' are good, so return true
    if p2 in p1 or p1 in p2:
        return True
    return phoneme_to_group.get(p1) == phoneme_to_group.get(p2)

def evaluate_accuracy():
    total = 0
    errors = defaultdict(list)
    similar_phonemes = defaultdict(list)
    exact_phonemes = defaultdict(list)

    for child in os.listdir(dataset_dir):
        child_path = os.path.join(dataset_dir, child)
        if not os.path.isdir(child_path):
            continue

        for character in os.listdir(child_path):
            character_path = os.path.join(child_path, character)
            expected_phoneme = char_to_phoneme.get(character.lower())

            if not expected_phoneme:
                print(f"[!] Skipping unknown character: {character}")
                continue

            total_files = len(os.listdir(character_path))
            count = 1
            for file in os.listdir(character_path):
                if not file.endswith(".wav"):
                    continue

                if '_th_' not in file:
                    continue
                total += 1
                file_path = os.path.join(character_path, file)

                print(f"Processing {file_path} ({count} of {total_files})...")
                with open(file_path, "rb") as f:
                    try:
                        response = requests.post(model_url, files={"audio": f})
                        response.raise_for_status()
                        actual_phoneme = response.json().get("phonemes", "").strip()
                    except Exception as e:
                        print(f"[!] Error processing {file_path}: {e}")
                        continue

                if actual_phoneme == expected_phoneme:
                    exact_phonemes[expected_phoneme].append((file_path, actual_phoneme))
                elif are_similar(actual_phoneme, expected_phoneme):
                    similar_phonemes[expected_phoneme].append((file_path, actual_phoneme))
                else:
                    errors[expected_phoneme].append((file_path, actual_phoneme))
                count += 1
    
    print(f"\nTotal files: {total}")
    # Get set of all phonemes in the exact/similar/error maps
    all_phonemes = set(exact_phonemes.keys()).union(similar_phonemes.keys()).union(errors.keys())
    for phoneme in all_phonemes:
        exact_count = len(exact_phonemes.get(phoneme, []))
        similar_count = len(similar_phonemes.get(phoneme, []))
        mismatches_count = len(errors.get(phoneme, []))
        print(f"=== Phoneme '{phoneme}' ===")
        print(f"Exact: {exact_count}")
        print(f"Similar: {similar_count}")
        print(f"Mismatches: {mismatches_count}")
        print(f"Accuracy: {(exact_count + similar_count)/total:.2%}\n")

        print("== Mismatches ==")
        for file_path, actual_phoneme in errors.get(phoneme, []):
            print(f"  - {file_path} => '{actual_phoneme}'")
    # combine exact and similar phonemes and errors into one list and make a dictionary with file_path as key and tuple of (expected_phoneme, 'similar'|'mismatch'|'exact') as value in one line
    phoneme_results = {
        file_path: (expected_phoneme, 'exact' if file_path in flatten(exact_phonemes.values()) else 'similar' if file_path in flatten(similar_phonemes.values()) else 'mismatch')
        for expected_phoneme, files in {**exact_phonemes, **similar_phonemes, **errors}.items()
        for file_path, _ in files
    }
    # Save phoneme_results for saving it for later use as a json file
    with open("phoneme_results.json", "w") as f:
        json.dump(phoneme_results, f, indent=4)
    print("\nPhoneme results saved to phoneme_results.json")

if __name__ == "__main__":
    evaluate_accuracy()
