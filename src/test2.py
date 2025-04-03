import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("./src/models/wav2vec2-finetuned-phonemes")
model = Wav2Vec2ForCTC.from_pretrained("./src/models/wav2vec2-finetuned-phonemes")

# ✅ Print vocab and PAD token info
print("Vocab:", processor.tokenizer.get_vocab())
print("PAD token ID:", processor.tokenizer.pad_token_id)

# Load audio
file_path = "multi_phoneme/ch/ch_rep_1.wav"
waveform, sample_rate = torchaudio.load(file_path)

print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate)
print("Max amplitude:", waveform.abs().max().item())

# Resample if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Preprocess audio
input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

# Run inference
with torch.no_grad():
    logits = model(input_values).logits

# ✅ Softmax for confidence check (optional)
probs = torch.nn.functional.softmax(logits, dim=-1)
top_probs, top_ids = torch.topk(probs, k=1, dim=-1)
print("Top token probabilities per timestep:", top_probs.squeeze())

# Decode predictions
predicted_ids = torch.argmax(logits, dim=-1)
print("Predicted IDs:", predicted_ids)
decoded = processor.decode(predicted_ids[0])
print("Decoded output:", decoded)

# Optional: token mapping
print("ID to token mapping:")
for idx in predicted_ids[0].tolist():
    print(f"{idx} => {processor.tokenizer.convert_ids_to_tokens(idx)}")
