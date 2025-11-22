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

# --- Data Models ---
class ChatRequest(BaseModel): 
    message: str
    history: list[dict] = [] # Accepts chat history

class ChatResponse(BaseModel): 
    reply: str
    sources: list[str]

# --- Setup Resources ---
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

# --- Helper Functions ---

def get_live_events(query: str):
    """Fetches relevant rows from Google Sheet based on query keywords."""
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
                    continue
        
        if not relevant_events: return None
        
        event_context = "Here are some relevant, upcoming events:\n"
        for event in relevant_events:
            event_context += f"- {event.get('Event Name')} on {event.get('Date')} ({event.get('Brief Description')})\n"
        return event_context
    except: return None

def perform_web_search(query: str):
    """Performs a live web search using DuckDuckGo."""
    print(f"  Performing Web Search for: {query}")
    try:
        results = DDGS().text(f"Marywood University {query}", max_results=3)
        if not results: return None, []
        search_context = "Web Search Results:\n"
        for r in results:
            search_context += f"- {r['title']}: {r['body']} (Source: {r['href']})\n"
        return search_context, [r['href'] for r in results]
    except Exception as e:
        print(f"Search error: {e}")
        return None, []

# --- Main Chat Handler ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    if not gemini_model: raise HTTPException(status_code=503, detail="AI Unavailable")
    print(f"Received query: {request.message}")

    # 1. Format Chat History (Last 6 turns to save context)
    conversation_history = ""
    for turn in request.history[-6:]: 
        role = "User" if turn['sender'] == "user" else "Maxis"
        clean_text = turn['text'].replace("\n", " ") 
        conversation_history += f"{role}: {clean_text}\n"

    # 2. Check Live Events
    event_context = get_live_events(request.message)
    if event_context:
        try:
            prompt = f"""
            You are Maxis.ai. 
            History: {conversation_history}
            User Question: {request.message}
            Event Data: {event_context}
            Task: Answer enthusiastically using the event data.
            """
            response = gemini_model.generate_content(prompt)
            return ChatResponse(reply=response.text, sources=[GOOGLE_SHEET_URL])
        except: pass

    # 3. RAG Search (Internal Docs)
    retrieved = collection.query(query_texts=[request.message], n_results=15, include=['documents', 'metadatas'])
    docs = retrieved.get('documents', [[]])[0]
    metas = retrieved.get('metadatas', [[]])[0]
    
    scored_docs = []
    if docs:
        pairs = [[request.message, d] for d in docs]
        scores = cross_encoder_model.predict(pairs)
        scored_docs = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

    # 4. Web Search (Always active for augmentation)
    web_text, web_sources = perform_web_search(request.message)
    
    # 5. Build Context
    sources = []
    context_parts = []

    # Add Top Internal Docs (Filter out absolute garbage, score > -2.0)
    top_docs = [d for s, d, m in scored_docs[:5] if s > -2.0]
    if top_docs:
        context_parts.append(f"--- TRUSTED CAMPUS KNOWLEDGE ---\n" + "\n\n".join(top_docs))
        for _, _, m in scored_docs[:5]:
            if m.get('source') and m.get('source') not in sources: sources.append(m.get('source'))

    # Add Web Docs
    if web_text:
        context_parts.append(f"--- LATEST WEB INFO ---\n{web_text}")
        for src in web_sources:
            if src not in sources: sources.append(src)

    full_context = "\n\n".join(context_parts)

    # 6. Generate Answer (The Prompt)
    prompt = f"""
    You are Maxis.ai, the spirited and knowledgeable AI mascot for Marywood University.
    
    RECENT CONVERSATION HISTORY:
    {conversation_history}

    KNOWLEDGE BASE:
    {full_context}

    USER QUESTION:
    {request.message}

    INSTRUCTIONS:
    1. **Be Specific:** If the user asks about athletics, LIST the sports. If they ask about majors, LIST the majors. Do not just say "we offer many sports."
    2. **Use Bullet Points or other Formatting:** For lists (teams, dates, requirements), use bullet points or other list formatting to make it readable. Use paragraphs and other text sectioning for explanations.
    3. **No Meta-Talk:** The user CANNOT see the "Knowledge Base" or "Context". 
       - NEVER say "As seen in the internal docs..." or "The context provided..."
       - Just say "Marywood offers..." or "I found that..."
    4. **Source Blending:** Seamlessly blend the "Trusted Campus Knowledge" (Policies/Facts) with "Web Info" (News/Scores).
    5. **Direct & Honest:** Extract the answer directly. If the exact list/fact is missing, say "I couldn't find the specific list in my records, but generally..." rather than making it up.
    6. **Formatting:** Use **bold** for emphasis. Use Markdown links `[Link Name](URL)`.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return ChatResponse(reply=response.text, sources=sources[:6])
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")