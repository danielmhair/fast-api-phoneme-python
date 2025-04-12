from fastapi import FastAPI, UploadFile, File
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer
import torchaudio
import torch
import shutil
import soundfile as sf

app = FastAPI()

processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


# Local Model Loading
# def load_model():
#     global processor, model
#     if processor is None or model is None:

#         processor = Wav2Vec2Processor.from_pretrained("./models/wav2vec2-finetuned-phonemes")
#         model = Wav2Vec2ForCTC.from_pretrained("./models/wav2vec2-finetuned-phonemes")
    

def predict_phonemes(file_path):
    load_model()

    speech_array, sampling_rate = sf.read(file_path)

    # Convert to mono if stereo
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    # Convert to float32 tensor
    speech_tensor = torch.from_numpy(speech_array).float()

    # Resample if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = resampler(speech_tensor)

    # Use processor to extract input features (NO padding)
    inputs = processor(speech_tensor, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    
    print("Predicted IDs:", predicted_ids)
    print("Decoded Output:", transcription)
    return transcription[0]


@app.post("/predict-phonemes")
async def predict_phoneme_endpoint(audio: UploadFile = File(...)):
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    phoneme_result = predict_phonemes(temp_audio_path)
    return {"phonemes": phoneme_result}
