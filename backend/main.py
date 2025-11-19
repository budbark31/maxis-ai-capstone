import os
import google.generativeai as genai
import chromadb
from duckduckgo_search import DDGS  # <--- NEW
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

def perform_web_search(query: str):
    print(f"  Performing Web Search for: {query}")
    try:
        results = DDGS().text(f"Marywood University {query}", max_results=3)
        if not results:
            return None
            
        search_context = "Web Search Results:\n"
        for r in results:
            search_context += f"- {r['title']}: {r['body']} (Source: {r['href']})\n"
        
        return search_context, [r['href'] for r in results]
    except Exception as e:
        print(f"Search failed: {e}")
        return None, []

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    if not gemini_model:
        raise HTTPException(status_code=503, detail="AI models are not available.")

    print(f"Received query: {request.message}")

    # --- PHASE 1: LIVE EVENTS (Google Sheets) ---
    event_context = get_live_events(request.message)
    if event_context:
        # ... (Keep your existing event logic here) ...
        pass 

    # --- PHASE 2: DATABASE SEARCH (RAG) ---
    retrieved_results = collection.query(
        query_texts=[request.message],
        n_results=15,
        include=['documents', 'metadatas']
    )
    retrieved_docs = retrieved_results.get('documents', [[]])[0]
    retrieved_metadatas = retrieved_results.get('metadatas', [[]])[0]

    # Re-Rank
    pairs = [[request.message, doc] for doc in retrieved_docs]
    scores = cross_encoder_model.predict(pairs)
    scored_docs = sorted(zip(scores, retrieved_docs, retrieved_metadatas), key=lambda x: x[0], reverse=True)
    
    # Filter for HIGH QUALITY matches only
    # (If the best match is weak, we should just go to the web)
    top_docs = [doc for score, doc, meta in scored_docs[:5] if score > 0.2] 
    
    sources = []
    context_text = ""
    
    # --- PHASE 3: THE DECISION ---
    
    if top_docs:
        print("  Found good database matches. Using RAG.")
        context_text = "\n\n".join(top_docs)
        for _, doc, meta in scored_docs[:5]:
            if meta.get('source') and meta.get('source') not in sources:
                sources.append(meta.get('source'))
                
    else:
        print("  Database matches are weak. Switching to Web Search...")
        web_text, web_sources = perform_web_search(request.message)
        if web_text:
            context_text = web_text
            sources = web_sources
        else:
            # If both fail, empty context will trigger the "I don't know" response
            context_text = ""

    # --- PHASE 4: GENERATE ANSWER ---
    prompt = f"""
    You are Maxis.ai, the friendly mascot of Marywood University.
    
    INSTRUCTIONS:
    1. Answer the user's question using the Context provided below.
    2. If the context is a "Web Search Result", explicitly mention that you found this info on the web.
    3. If the context is empty, politely say you don't know.
    4. Be helpful, encouraging, and concise.

    Context:
    {context_text}

    User's Question:
    {request.message}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return ChatResponse(reply=response.text, sources=sources)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")