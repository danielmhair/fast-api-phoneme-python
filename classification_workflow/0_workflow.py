from w1_prepare_wav_files import prepare_wav_files
from w2_extract_embeddings_for_phonemes import extract_embeddings_for_phonemes
from w3_embeddings import embeddings
from w4_batch_test_phonemes import batch_test_phonemes

def main():
    # Prepare the WAV files
    prepare_wav_files()

    # Extract embeddings for phonemes
    extract_embeddings_for_phonemes()

    # Extract embeddings
    embeddings()

    # Batch test phonemes
    batch_test_phonemes()

if __name__ == "__main__":
    main()
    