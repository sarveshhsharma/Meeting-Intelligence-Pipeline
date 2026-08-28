import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Import our schema
from backend.models.schemas import MeetingSummary

def generate_summary_and_tasks(transcript_text: str) -> MeetingSummary:
    """
    Takes a raw string transcript, passes it to GPT-4o-mini, and forces
    the output into our strict MeetingSummary JSON structure.
    """
    if not transcript_text or len(transcript_text.strip()) == 0:
        raise ValueError("Transcript text is empty. Cannot generate summary.")

    # 1. Initialize the parser with our Pydantic class
    parser = PydanticOutputParser(pydantic_object=MeetingSummary)
    
    # 2. Build the prompt. 
    # Notice we inject the parser's instructions directly into the prompt.
    prompt = PromptTemplate(
        template="""
        You are an expert enterprise business analyst. Review the following meeting transcript 
        and extract the key intelligence accurately.
        
        {format_instructions}
        
        --- MEETING TRANSCRIPT ---
        {transcript}
        """,
        input_variables=["transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 3. Initialize the Large Language Model
    # temperature=0 ensures the AI acts deterministically and factually, not creatively
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini",
        openai_api_key=os.environ.get("OPENAI_API_KEY") 
    )
    
    # 4. Chain the components together (Prompt -> Model -> Parser)
    chain = prompt | llm | parser
    
    # 5. Execute the chain and return the resulting MeetingSummary object
    print("Sending transcript to LLM for intelligence extraction...")
    structured_data = chain.invoke({"transcript": transcript_text})
    
    return structured_data

#########NEW ADDITION TO MAKE RAG BASED ########### 
def extract_relevant_info_from_chunk(chunk: str, question: str) -> str:
    """
    MAP STEP: Analyzes a single chunk of text and extracts only the 
    information relevant to the user's question.
    """
    prompt = PromptTemplate(
        template="""
        You are a meticulous data extractor. Review the following text chunk from a meeting.
        Extract and summarize ONLY the information that is relevant to answering the user's question.
        If the chunk contains no relevant information, output exactly: "No relevant information."
        
        TEXT CHUNK:
        {chunk}
        
        USER QUESTION:
        {question}
        """,
        input_variables=["chunk", "question"]
    )
    
    # We can use a fast, cheap model here since it's just extracting
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    chain = prompt | llm
    
    response = chain.invoke({"chunk": chunk, "question": question})
    return response.content.strip()

def generate_rag_answer(question: str, context: str) -> str:
    """Uses GPT-4o-mini to answer a question based ONLY on the provided context."""
    prompt = PromptTemplate(
        template="""
        You are a helpful meeting assistant. Answer the user's question using ONLY the provided meeting transcript context.
        If the answer is not contained in the context, say "I cannot find the answer to this in the meeting transcript."
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        """,
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "question": question})
    return response.content


# #################DUMMY CODE ######
# from backend.models.schemas import MeetingSummary, ActionItem


# def generate_summary_and_tasks(transcript_text: str) -> MeetingSummary:
#     """
#     Dummy implementation of the LLM.

#     Instead of calling OpenAI, this function returns hardcoded
#     meeting intelligence so the rest of the application
#     (FastAPI, frontend, vector database, etc.) can be tested.
#     """

#     if not transcript_text or len(transcript_text.strip()) == 0:
#         raise ValueError("Transcript text is empty. Cannot generate summary.")

#     print("Using Dummy LLM...")

#     return MeetingSummary(
#         executive_summary=(
#             "The meeting focused on reviewing the progress of the Enterprise "
#             "Meeting Intelligence Pipeline. Team members discussed the current "
#             "status of the backend, transcription service, vector database, and "
#             "frontend integration.\n\n"
#             "The team agreed to continue using FastAPI for the backend, Whisper "
#             "for transcription, and a vector database for semantic search. "
#             "Everyone was aligned on completing testing before the next demo."
#         ),

#         key_decisions=[
#             "Continue development using FastAPI.",
#             "Use Whisper as the transcription engine.",
#             "Store meeting embeddings in the vector database.",
#             "Integrate the frontend with the backend before deployment.",
#             "Perform end-to-end testing this week."
#         ],

#         action_items=[
#             ActionItem(
#                 task="Complete backend authentication module.",
#                 assignee="Alice",
#                 due_date="Next Monday",
#                 priority="High"
#             ),
#             ActionItem(
#                 task="Connect the frontend with the meeting processing API.",
#                 assignee="Bob",
#                 due_date="Wednesday",
#                 priority="High"
#             ),
#             ActionItem(
#                 task="Test semantic search functionality.",
#                 assignee="Charlie",
#                 due_date="Friday",
#                 priority="Normal"
#             ),
#             ActionItem(
#                 task="Prepare the project demo presentation.",
#                 assignee="David",
#                 due_date="End of the week",
#                 priority="Low"
#             ),
#             ActionItem(
#                 task="Review API documentation and update missing endpoints.",
#                 assignee=None,
#                 due_date=None,
#                 priority="Normal"
#             )
#         ],

#         overall_sentiment="Collaborative"
#     )

# def extract_relevant_info_from_chunk(chunk: str, question: str) -> str:
#     """
#     DUMMY FUNCTION: Simulates the MAP step. 
#     Pretends to analyze a single chunk and extract relevant info.
#     """
#     print(f"🛠️ [DUMMY] Simulating Map Step extraction for question: '{question}'...")
    
#     # We simulate that sometimes a chunk has no useful info, 
#     # but most of the time it returns a mock extracted fact.
#     if len(chunk) < 10: 
#         return "No relevant information."
        
#     return f"Mock extracted detail from chunk: The team discussed aspects related to '{question}'."

# def generate_rag_answer(question: str, context: str) -> str:
#     """
#     DUMMY FUNCTION: Simulates the RAG Chat response.
#     """
#     print(f"🛠️ [DUMMY] Simulating RAG answer for question: '{question}'...")
#     return f"This is a dummy response. Based on the mock transcript, Sarah is responsible for updating the dashboard to blue by next Tuesday."