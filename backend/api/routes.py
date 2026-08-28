from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from typing import Dict, Any

# Import our AI services
from backend.services.transcription import transcribe_audio
from backend.services.llm_engine import generate_summary_and_tasks
from backend.services.vector_store import save_meeting_to_db, search_meetings, get_all_meetings, search_specific_meeting
from backend.services.llm_engine import generate_rag_answer, extract_relevant_info_from_chunk

router = APIRouter()

# Define the upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../../data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SearchQuery(BaseModel):
    query: str

@router.post("/process-meeting")
async def process_meeting(file: UploadFile = File(...)):
    """
    Full Pipeline: Upload Audio -> Transcribe -> Extract Tasks -> Save to Vector DB
    """
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload mp3 or wav.")

    # 1. Save the file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Transcribe the audio
        transcription_result = transcribe_audio(file_path)
        
        # 3. Pass transcript to LLM for intelligence extraction
        intelligence = generate_summary_and_tasks(transcription_result.transcript_text)
        
        # 4. Save to Vector Database for semantic search later
        save_meeting_to_db(
            meeting_id=file.filename,
            transcript=transcription_result.transcript_text,
            summary=intelligence.model_dump()
        )

        # 5. Return the full structured payload to the frontend
        return {
            "status": "success",
            "transcription": transcription_result,
            "intelligence": intelligence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_history(query: SearchQuery):
    """
    Search past meetings using natural language.
    """
    try:
        results = search_meetings(query.query)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
########### ADDITIONAL FOR RAG BASED 
class MeetingChatQuery(BaseModel):
    query: str
    meeting_id: str

@router.get("/meetings")
async def list_meetings():
    """Endpoint to populate the Streamlit dropdown menu."""
    try:
        meetings = get_all_meetings()
        return {"status": "success", "meetings": meetings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/meeting-chat")
async def chat_with_meeting(payload: MeetingChatQuery):
    """The Map-Reduce RAG endpoint for a specific meeting."""
    try:
        # 1. Retrieve the most relevant chunks from ChromaDB
        search_results = search_specific_meeting(payload.query, payload.meeting_id)
        documents = search_results.get("documents", [[]])[0]
        
        if not documents:
            return {"status": "success", "answer": "I couldn't find any relevant discussion about that in this meeting."}
        
        # 2. MAP STEP: Process each chunk separately against the query
        print(f"Processing {len(documents)} chunks individually...")
        chunk_summaries = []
        
        for idx, chunk in enumerate(documents):
            # Extract only the useful info from this specific chunk
            summary = extract_relevant_info_from_chunk(chunk, payload.query)
            
            # Filter out chunks that had nothing useful to save tokens
            if "No relevant information" not in summary:
                chunk_summaries.append(f"Source {idx+1}: {summary}")
        
        # If all chunks returned "No relevant information", fail gracefully
        if not chunk_summaries:
            return {"status": "success", "answer": "I found some transcripts, but none of them contained the answer to your question."}
        
        # 3. Combine the condensed, highly-relevant summaries into one context string
        context_string = "\n\n".join(chunk_summaries)
        
        # 4. REDUCE STEP: Generate the final answer using the condensed context
        print("Generating final RAG answer from condensed context...")
        answer = generate_rag_answer(payload.query, context_string)
        
        return {"status": "success", "answer": answer, "context_used": documents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))