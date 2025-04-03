from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os

app = FastAPI()
model = None

def load_model():
    """
    Load the Whisper model once (lazy load). 
    Choose from: 'tiny', 'base', 'small', 'medium', 'large'.
    """
    global model
    if model is None:
        model = whisper.load_model("large")  # 'tiny' is faster but less accurate

@app.post("/predict-phonemes")
async def predict_phoneme_endpoint(audio: UploadFile = File(...)):
    """
    Accepts an uploaded audio file under the field "audio" 
    and returns a transcription (labelled "phonemes" to match your original).
    """
    load_model()

    # Write uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
        content = await audio.read()
        tmp.write(content)

    try:
        # Use Whisper to transcribe the audio
        result = model.transcribe(tmp_name)
        transcription = result.get("text", "").strip()
    finally:
        # Clean up the temp file
        os.remove(tmp_name)

    # Return in a field called "phonemes" (so your code calling this won't break).
    print("Transcription:", transcription)
    return {"phonemes": transcription}
