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
from duckduckgo_search import DDGS 
import re

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

class ChatRequest(BaseModel): 
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel): 
    reply: str
    sources: list[str]

# --- Setup ---
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
    print("Services loaded successfully.")
except Exception as e:
    print(f"FATAL: Service load error: {e}")
    gemini_model = None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Helpers ---
def get_live_events(query: str):
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
                    if event_dt > now: relevant_events.append(event)
                except: continue
        if not relevant_events: return None
        event_context = "Here are some relevant, upcoming events:\n"
        for event in relevant_events:
            event_context += f"- {event.get('Event Name')} on {event.get('Date')} ({event.get('Brief Description')})\n"
        return event_context
    except: return None

# --- NEW: Smart Query Expansion ---
def perform_web_search(user_query: str):
    # 1. Clean the query
    clean_query = re.sub(r'\b(what|is|the|last|recent|latest|score|game|result|did|they|win|lose)\b', '', user_query, flags=re.IGNORECASE).strip()
    
    # 2. Basic Search Query
    if "marywood" not in clean_query.lower():
        search_query = f"Marywood University {clean_query}"
    else:
        search_query = clean_query

    # 3. CRITICAL: Inject "Current Season" Year
    # If users ask about sports, we force the search engine to look for "2025-26"
    # This prevents it from finding the "2024-25 Season Recap" page.
    now = datetime.now(TIMEZONE)
    if any(x in user_query.lower() for x in ['score', 'game', 'result', 'schedule', 'record', 'won', 'lost']):
        # If it's Aug-Dec, the season is Year-(Year+1) (e.g. 2025-26)
        # If it's Jan-July, the season is (Year-1)-Year (e.g. 2025-26)
        if now.month >= 8:
            season_str = f"{now.year}-{str(now.year+1)[-2:]}"
        else:
            season_str = f"{now.year-1}-{str(now.year)[-2:]}"
        
        search_query += f" {season_str} schedule results"

    print(f"  Performing Smart Web Search for: '{search_query}'")
    
    try:
        # Fetch 8 results to ensure we hit the schedule page
        results = DDGS().text(search_query, max_results=8) 
        if not results: return None, []
        
        search_context = "Web Search Results:\n"
        for r in results:
            search_context += f"- {r['title']}: {r['body']} (Source: {r['href']})\n"
        
        return search_context, [r['href'] for r in results]
    except Exception as e:
        print(f"Search failed: {e}")
        return None, []

@app.get("/api/stats")
async def get_stats():
    try:
        count = collection.count()
        return {"status": "online", "documents_indexed": count, "model": GEMINI_MODEL_NAME}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# --- Chat Handler ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    if not gemini_model: raise HTTPException(status_code=503, detail="AI Unavailable")
    print(f"Received query: {request.message}")

    # 1. Time Context
    current_time = datetime.now(TIMEZONE)
    current_time_str = current_time.strftime("%A, %B %d, %Y")
    
    # 2. History
    conversation_history = ""
    for turn in request.history[-6:]: 
        role = "User" if turn['sender'] == "user" else "Maxis"
        clean_text = turn['text'].replace("\n", " ") 
        conversation_history += f"{role}: {clean_text}\n"

    # 3. Live Events
    event_context = get_live_events(request.message)
    if event_context:
        try:
            prompt = f"You are Maxis.ai. Time: {current_time_str}\nHistory: {conversation_history}\nUser: {request.message}\nData: {event_context}\nTask: Answer enthusiastically."
            response = gemini_model.generate_content(prompt)
            return ChatResponse(reply=response.text, sources=[GOOGLE_SHEET_URL])
        except: pass

    # 4. RAG Search
    retrieved = collection.query(query_texts=[request.message], n_results=15, include=['documents', 'metadatas'])
    docs = retrieved.get('documents', [[]])[0]
    metas = retrieved.get('metadatas', [[]])[0]
    
    scored_docs = []
    if docs:
        pairs = [[request.message, d] for d in docs]
        scores = cross_encoder_model.predict(pairs)
        scored_docs = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

    # 5. Web Search
    web_text, web_sources = perform_web_search(request.message)
    
    # 6. Build Context
    sources = []
    context_parts = []

    # Add Top Internal Docs
    top_docs = [d for s, d, m in scored_docs[:5] if s > -2.0]
    if top_docs:
        context_parts.append(f"--- INTERNAL HANDBOOKS (STATIC POLICY DATA) ---\n" + "\n\n".join(top_docs))
        for _, _, m in scored_docs[:5]:
            if m.get('source') and m.get('source') not in sources: sources.append(m.get('source'))

    # Add Web Docs
    if web_text:
        context_parts.append(f"--- WEB SEARCH RESULTS (LIVE NEWS DATA) ---\n{web_text}")
        for src in web_sources:
            if src not in sources: sources.append(src)

    full_context = "\n\n".join(context_parts)

    # 7. Generate Answer
    prompt = f"""
    You are Maxis.ai, the spirited AI mascot for Marywood University.
    
    TODAY'S DATE: {current_time_str}
    
    HISTORY:
    {conversation_history}

    KNOWLEDGE BASE:
    {full_context}

    USER QUESTION:
    {request.message}

    INSTRUCTIONS:
    1. **Time Awareness:** You MUST contextualize all information relative to TODAY'S DATE.
       - If the user asks for "current", "next", "latest", or "recent" information (whether sports, academic dates, or events), prioritize the data closest to TODAY.
       - Discard outdated schedules (e.g., last year's calendar) in favor of the current or upcoming academic year.
       - Example: If today is Dec 2025, a "Fall 2025" document is relevant, but a "Spring 2025" document is past.
    2. **Trust Hierarchy:**
       - For **DYNAMIC INFO** (News, Sports, Events, Deadlines): Trust "WEB SEARCH RESULTS" or the most recent date found.
       - For **STATIC POLICY** (Rules, grading, housing guidelines): Trust "INTERNAL HANDBOOKS".
    3. **Be Specific:** List scores, opponents, specific dates, and deadlines. Avoid vague summaries.
    4. **No Meta-Talk:** Do not mention "internal docs", "context", or "training data".
    5. **Formatting:** Use **bold** for key terms and bullet points for lists. Use Markdown links `[Link Name](URL)`.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return ChatResponse(reply=response.text, sources=sources[:6])
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")