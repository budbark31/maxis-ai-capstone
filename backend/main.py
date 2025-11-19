import os
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from urllib.parse import urlparse
import pytz

# --- Configuration ---
load_dotenv()
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
COLLECTION_NAME = "marywood_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dvaZu97P9-QSzCmKBBcA_RshyqRAjU67yZ1398EFei4/edit?usp=sharing"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'credentials.json')
TIMEZONE = pytz.timezone('America/New_York')

# --- Pydantic Models (Unchanged) ---
class ChatRequest(BaseModel): message: str
class ChatResponse(BaseModel): reply: str; sources: list[str]

# --- Global Resources ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_url(GOOGLE_SHEET_URL)
    sheet = spreadsheet.sheet1
    print("Models, DB, and Google Sheets client loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load models or connect to services: {e}")
    gemini_model = None

# --- FastAPI App (Unchanged) ---
app = FastAPI()
origins = ["http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Helper Function for Live Events (Unchanged) ---
def get_live_events(query: str):
    print("Checking for live events...")
    try:
        records = sheet.get_all_records()
        now = datetime.now(TIMEZONE)
        
        relevant_events = []
        for event in records:
            if any(keyword.lower() in str(event.values()).lower() for keyword in query.lower().split()):
                try:
                    event_dt_str = f"{event.get('Date')} {event.get('End Time')}"
                    try:
                        event_dt = TIMEZONE.localize(datetime.strptime(event_dt_str, '%m/%d/%Y %I:%M:%S %p'))
                    except ValueError:
                         event_dt = TIMEZONE.localize(datetime.strptime(event_dt_str, '%m/%d/%Y %H:%M:%S'))

                    if event_dt > now:
                        relevant_events.append(event)
                except (ValueError, TypeError):
                    print(f"Could not parse date/time for event: {event.get('Event Name')} - '{event.get('Date')} {event.get('End Time')}'")
                    continue
        
        if not relevant_events:
            return None

        event_context = "Here are some relevant, upcoming events:\n"
        for event in relevant_events:
            name = event.get('Event Name', 'N/A')
            date = event.get('Date', 'N/A')
            start_time = event.get('Start Time', 'N/A')
            end_time = event.get('End Time', 'N/A')
            location = event.get('Location', 'N/A')
            description = event.get('Brief Description', 'N/A')
            event_context += f"- Event: {name}, Date: {date}, Time: {start_time} to {end_time}, Location: {location}, Description: {description}\n"
        
        return event_context

    except gspread.exceptions.APIError as e:
         print(f"Google Sheets API Error: {e}. Check sharing permissions and API key.")
         return None
    except Exception as e:
        print(f"Error fetching from Google Sheet: {e}")
        return None

# --- Chat Handler Logic (changed) ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    if not gemini_model:
        raise HTTPException(status_code=503, detail="AI models are not available.")

    print(f"Received query: {request.message}")

    # 1. Try Live Events
    event_context = get_live_events(request.message)
    if event_context:
        prompt = f"""
        You are Maxis.ai, the spirited mascot and AI assistant of Marywood University.
        User Question: {request.message}
        
        Real-Time Event Data:
        {event_context}
        
        Task: Answer the question enthusiastically using the event data above.
        """
        try:
            response = gemini_model.generate_content(prompt)
            return ChatResponse(reply=response.text, sources=[GOOGLE_SHEET_URL])
        except Exception:
            pass # Fallback to RAG if this fails
    
    # 2. RAG Search
    retrieved_results = collection.query(
        query_texts=[request.message],
        n_results=15, # Fetch more chunks so Gemini has more context to summarize
        include=['documents', 'metadatas']
    )
    retrieved_docs = retrieved_results.get('documents', [[]])[0]
    retrieved_metadatas = retrieved_results.get('metadatas', [[]])[0]

    # Rank and Filter
    pairs = [[request.message, doc] for doc in retrieved_docs]
    scores = cross_encoder_model.predict(pairs)
    scored_docs = sorted(zip(scores, retrieved_docs, retrieved_metadatas), key=lambda x: x[0], reverse=True)
    
    # Keep top 5 most relevant chunks
    top_docs = [doc for score, doc, meta in scored_docs[:5] if score > 0] # Only keep positive relevance
    
    # Extract Sources
    seen_srcs = []
    for _, doc, meta in scored_docs[:5]:
        if meta.get('source') and meta.get('source') not in seen_srcs:
            seen_srcs.append(meta.get('source'))

    context_text = "\n\n".join(top_docs)
    
    # 3. The "Natural" Prompt
    prompt = f"""
    You are Maxis.ai, the friendly, helpful, and spirited mascot of Marywood University.
    Your goal is to be a helpful guide, not a robot reading from a script.

    INSTRUCTIONS:
    1. **Synthesize, Don't Quote:** Read the Context below, understand it, and explain the answer to the user in your own words. Do not just copy-paste sentences.
    2. **Be Conversational:** If the user greets you, greet them back warmly before addressing their question.
    3. **Honesty:** If the context doesn't contain the answer, admit it gracefully. Say something like, "I checked my university documents, but I couldn't find that specific detail."
    4. **Formatting:** Use bullet points or short paragraphs to make the answer easy to read.

    Context from University Documents:
    {context_text}

    User's Question:
    {request.message}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return ChatResponse(reply=response.text, sources=seen_srcs)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")
    
    print("No live events found or event-based generation failed. Performing RAG search.")
    retrieved_results = collection.query(
        query_texts=[request.message],
        n_results=10,
        include=['documents', 'metadatas']
    )
    retrieved_docs = retrieved_results.get('documents', [[]])[0]
    retrieved_metadatas = retrieved_results.get('metadatas', [[]])[0]
    
    if not retrieved_docs:
        return ChatResponse(reply="I'm sorry, I couldn't find any information on that topic.", sources=[])

    print(f"Re-ranking {len(retrieved_docs)} documents...")
    pairs = [[request.message, doc] for doc in retrieved_docs]
    scores = cross_encoder_model.predict(pairs)
    
    scored_docs = sorted(zip(scores, retrieved_docs, retrieved_metadatas), key=lambda x: x[0], reverse=True)
    
    top_k_docs = [doc for score, doc, meta in scored_docs[:5]]

    # Prefer official marywood.edu sources over athletics site and cached_document
    def domain_score(url: str) -> int:
        if not url: return 0
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            netloc = url.lower()
        if 'marywood.edu' in netloc:
            return 100
        if 'marywoodpacers.com' in netloc:
            return 10
        if url == 'cached_document':
            return 1
        return 50

    seen_srcs = []
    for _, doc, meta in scored_docs[:20]:
        src = None
        try:
            src = meta.get('source') if isinstance(meta, dict) else None
        except Exception:
            src = None
        if not src:
            continue
        if src in seen_srcs:
            continue
        seen_srcs.append(src)

    # sort seen sources by domain priority and return top 5
    source_urls = sorted(seen_srcs, key=lambda u: domain_score(u), reverse=True)[:5]

    context = "\n\n".join(top_k_docs)
    
    prompt = f"""
    You are Maxis.ai, the friendly and helpful mascot of Marywood University. Your tone is encouraging and clear.
    Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, say "I'm sorry, I don't have information on that topic based on the documents I have."

    Context:
    {context}

    User's Question:
    {request.message}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return ChatResponse(reply=response.text, sources=source_urls)
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate RAG response.")

