import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Create or connect to a local database folder
# We use the data/chromadb folder you set up in Phase 1
db_path = os.path.join(os.path.dirname(__file__), "../../data/chromadb")
chroma_client = chromadb.PersistentClient(path=db_path)

# We use OpenAI's default embedding model to turn text into vectors
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Get or create a "collection" (like a table in SQL) for our meetings
collection = chroma_client.get_or_create_collection(
    name="meeting_transcripts", 
    embedding_function=openai_ef
)

def save_meeting_to_db(meeting_id: str, transcript: str, summary: dict):
    """
    Chunks the transcript and saves it into the vector database.
    """
    print(f"Chunking and saving meeting {meeting_id} to vector database...")
    
    # 1. Initialize the Splitter
    # chunk_size: How many characters per chunk (1000 chars is roughly 200 words)
    # chunk_overlap: How many characters to overlap so we don't cut a thought in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len,
    )
    
    # 2. Break the transcript into a list of strings
    chunks = text_splitter.split_text(transcript)
    print(f"Broke transcript into {len(chunks)} chunks.")
    
    # 3. Create unique IDs and duplicate the metadata for every chunk
    # ChromaDB needs every single chunk to have its own unique ID
    chunk_ids = [f"{meeting_id}_chunk_{i}" for i in range(len(chunks))]
    
    # We attach the same filename and summary to every chunk so we can filter by it later
    metadatas = [{"filename": meeting_id, "summary": str(summary)} for _ in chunks]
    
    # 4. Save the chunks to the database
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )

def search_meetings(query: str, n_results: int = 3):
    """
    Searches past meetings using natural language.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

########### NEW ADDITION #######(for RAG BASED)
def get_all_meetings():
    """Returns a list of unique meeting filenames stored in the database."""
    # Fetch all metadata records from ChromaDB
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])
    
    # Extract unique filenames using a set
    unique_meetings = set()
    for meta in metadatas:
        if meta and "filename" in meta:
            unique_meetings.add(meta["filename"])
            
    return list(unique_meetings)

def search_specific_meeting(query: str, meeting_id: str, n_results: int = 3):
    """Searches vector space, but FILTERS strictly to one specific meeting."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"filename": meeting_id} # This tells ChromaDB to ignore other files
    )
    return results