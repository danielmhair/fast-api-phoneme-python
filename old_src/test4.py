# Sanity check for tokenizer behavior
from transformers import Wav2Vec2Processor

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("./src/models/wav2vec2-finetuned-phonemes-2")
# Load your existing processor/tokenizer
print("Vocab:", processor.tokenizer.get_vocab())

# Try encoding and decoding a known phoneme with bars
phoneme = "f"
input_ids = processor.tokenizer(phoneme).input_ids
decoded = processor.tokenizer.decode(input_ids)

print(f"Original: {phoneme}")
print(f"Encoded: {input_ids}")
print(f"Decoded: {decoded}")
