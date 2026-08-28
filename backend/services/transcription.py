import whisper
import os
import time

# Import our exact schema to enforce data structure
from backend.models.schemas import TranscriptionResponse

print("Loading Whisper model into memory... (This may take a moment)")
# Load the model globally so it stays in RAM for fast API responses
MODEL = whisper.load_model("base")

def transcribe_audio(file_path: str) -> TranscriptionResponse:
    """
    Takes a path to an audio file, transcribes it using OpenAI's Whisper,
    and returns a structured Pydantic object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find audio file at: {file_path}")

    print(f"Starting transcription for {file_path}...")
    start_time = time.time()

    # The actual AI execution
    result = MODEL.transcribe(file_path)
    
    end_time = time.time()
    print(f"Transcription finished in {round(end_time - start_time, 2)} seconds.")

    # Calculate approximate duration based on the audio segments
    segments = result.get('segments', [])
    duration = sum([seg['end'] - seg['start'] for seg in segments]) if segments else 0.0

    # Return the data locked into our strict Pydantic schema
    return TranscriptionResponse(
        filename=os.path.basename(file_path),
        transcript_text=result["text"].strip(),
        duration_seconds=round(duration, 2)
    )