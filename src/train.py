import os
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class CustomCTCCollator:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace pad_token_id with -100 so CTC loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id,
            -100
        )

        batch["labels"] = labels
        return batch


# Load and prepare CSV dataset
data_df = pd.read_csv("multi_dataset.csv")
dataset = Dataset.from_pandas(data_df)

# Load vocab and create tokenizer + feature extractor
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

# Combine into processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained("./src/models/wav2vec2-finetuned-phonemes")
def prepare_dataset(batch):
    MIN_INPUT_LENGTH = 3000  # ~0.2 sec at 16kHz
    

    speech_array, sampling_rate = torchaudio.load(batch["path"])
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    waveform = resampler(speech_array).squeeze()

    # Safeguard in case waveform is empty
    if waveform.numel() == 0:
        print(f"⚠️ Skipping empty waveform: {batch['path']}")
        return None

    input_values = processor(waveform.numpy(), sampling_rate=16000).input_values[0]

    # Corrected: Check 1D length directly
    if input_values.shape[0] < MIN_INPUT_LENGTH:
        return None  # Skip short sample

    batch["input_values"] = input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["phoneme"]).input_ids

    # Optional debug info
    # print("Phoneme:", batch["phoneme"])
    # print("Label IDs:", processor.tokenizer(batch["phoneme"]).input_ids)

    return batch

import torchaudio
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: x is not None)

# Define model
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    mask_time_length=2,  # 👈 try 2 since most files can be less than ~0.3s
)

# Training args
training_args = TrainingArguments(
    output_dir="./src/models/wav2vec2-finetuned-phonemes",
    group_by_length=True,
    per_device_train_batch_size=8,
    evaluation_strategy="no",  # You can later change this to "steps" or "epoch"
    save_strategy="epoch",
    num_train_epochs=5,  # Try more epochs for better learning
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    logging_steps=20,  # More frequent logs
    learning_rate=1e-4,
    warmup_steps=10,
)

data_collator = CustomCTCCollator(processor=processor, padding=True)

# Define Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./src/models/wav2vec2-finetuned-phonemes")
