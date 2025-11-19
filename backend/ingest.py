import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
from io import BytesIO
import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
load_dotenv()

# 1. PRIORITIZED LINKS (The crawler will visit these FIRST)
START_URLS = [
    # --- ACADEMICS & ADMIN ---
    "https://www.marywood.edu/registrar",               # Class registration, transcripts
    "https://www.marywood.edu/academics/calendar",      # Important dates
    "https://www.marywood.edu/academics/library",       # Research help, hours
    "https://www.marywood.edu/policy/handbooks/students/", # Rules & Regulations
    
    # --- DAILY STUDENT LIFE ---
    "https://www.marywood.edu/life-at-mu/dining",       # Menus, hours, locations
    "https://www.marywood.edu/life-at-mu/campus-life/residential/", # Housing info
    "https://www.marywood.edu/safety",                  # Campus safety, parking
    "https://marywoodpacers.com/",                      # Athletics schedules/scores

    # --- SUPPORT SERVICES (Crucial for chatbots) ---
    "https://www.marywood.edu/depts/IT/faq",            # Wi-Fi, passwords, tech support
    "https://www.marywood.edu/academics/success/career-center/", # Internships, jobs
    "https://www.marywood.edu/life-at-mu/student-experience/counseling/", # Mental health resources
    "https://www.marywood.edu/affordability/resources/", # Financial aid, scholarships

    # --- GENERAL ---
    "https://www.marywood.edu/contact",                 # Phone directory
    "https://www.marywood.edu/about/"                   # General university info
]

ALLOWED_DOMAINS = ["marywood.edu", "marywoodpacers.com"]

# 2. IGNORE PATTERNS (Refined for less junk)
IGNORE_PATTERNS = [
    "login", "signup", "javascript:", "mailto:", "tel:",
    "/boxscore/", "/stats/", "/history/", "/archives/", # Ignore old sports stats
    "/roster/", "/players/", # Ignore individual player bios (too much noise)
]

# Automatically add years 2000-2022 to ignore list
for year in range(2000, 2023):
    IGNORE_PATTERNS.append(str(year))

# 3. SAFETY LIMITS (Increased for real crawling)
MAX_PAGES_TO_CRAWL = 5000  # Increased from 300 to 5000
MAX_WORKERS = 10           # Increased speed (10 parallel threads)

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'visited_urls.json')
COLLECTION_NAME = "marywood_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MaxisAI-Capstone/1.0'}

os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper Functions ---
def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def is_ignored(url: str) -> bool:
    """Check if URL contains any ignored patterns."""
    for pattern in IGNORE_PATTERNS:
        if pattern in url.lower():
            return True
    return False

def get_cached_content(url: str) -> str | None:
    filename = get_url_hash(url)
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def cache_content(url: str, content: str):
    filename = get_url_hash(url)
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def fetch_url(url: str):
    """Fetches a single URL and returns the result data."""
    result = {
        "url": url,
        "text": "",
        "title": "Untitled",
        "new_links": [],
        "success": False,
        "is_pdf": False
    }
    
    # Check cache first
    cached_text = get_cached_content(url)
    if cached_text:
        # Quick parse to find links even in cached HTML
        if not url.lower().endswith(".pdf"):
            try:
                soup = BeautifulSoup(cached_text, "html.parser")
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(url, link['href']).split('#')[0]
                    if any(d in abs_url for d in ALLOWED_DOMAINS) and not is_ignored(abs_url):
                        result["new_links"].append(abs_url)
                result["title"] = soup.title.string.strip() if soup.title else "Cached"
            except: pass
        
        result["text"] = cached_text
        result["success"] = True
        return result

    # Live Fetch
    print(f"  [Fetching] {url}")
    try:
        response = requests.get(url, timeout=10, headers=HEADERS)
        if response.status_code != 200:
            return result
        
        final_url = response.url
        result["url"] = final_url 
        content_type = response.headers.get('content-type', '').lower()

        if 'application/pdf' in content_type or final_url.lower().endswith('.pdf'):
            result["is_pdf"] = True
            result["title"] = os.path.basename(final_url)
            with BytesIO(response.content) as f:
                reader = pypdf.PdfReader(f)
                text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
                result["text"] = text
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            result["title"] = soup.title.string.strip() if soup.title else "Untitled"
            main = soup.find('main') or soup.find('article') or soup.body
            if main:
                result["text"] = main.get_text(separator="\n", strip=True)
                # Find links
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(final_url, link['href']).split('#')[0]
                    if any(d in abs_url for d in ALLOWED_DOMAINS) and not is_ignored(abs_url):
                        result["new_links"].append(abs_url)
            
            cache_content(final_url, response.text)

        if result["text"]:
            if result["is_pdf"]: cache_content(final_url, result["text"])
            result["success"] = True

    except Exception as e:
        print(f"  [Error] {url}: {e}")
    
    return result

def scrape_and_crawl():
    """Parallel crawler with Priority Queue."""
    print(f"--- Starting Fast Crawl (Max {MAX_PAGES_TO_CRAWL} pages) ---")
    
    # Load visited state
    visited = set()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: visited = set(json.load(f))
        except: pass

    queue = deque(START_URLS)
    processed_docs = []
    count = 0

    while queue and count < MAX_PAGES_TO_CRAWL:
        batch_urls = []
        while len(batch_urls) < MAX_WORKERS and queue:
            u = queue.popleft()
            if u not in visited:
                visited.add(u)
                batch_urls.append(u)
        
        if not batch_urls: break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(fetch_url, url): url for url in batch_urls}
            
            for future in as_completed(future_to_url):
                res = future.result()
                if res["success"] and res["text"]:
                    count += 1
                    doc_id = get_url_hash(res["url"])
                    processed_docs.append({
                        "doc_id": doc_id,
                        "text": res["text"],
                        "source": res["url"],
                        "title": res["title"]
                    })
                    
                    for link in res["new_links"]:
                        if link not in visited:
                            queue.append(link)
        
        if count % 50 == 0:
            print(f"--- Progress: {count}/{MAX_PAGES_TO_CRAWL} pages processed ---")
            with open(STATE_FILE, 'w') as f: json.dump(list(visited), f)

    with open(STATE_FILE, 'w') as f: json.dump(list(visited), f)
    return processed_docs

def main():
    print("--- Starting Ingestion ---")
    new_docs = scrape_and_crawl()
    if not new_docs:
        print("No new documents found.")
        return

    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    )

    print(f"Embedding {len(new_docs)} documents...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    
    for doc in new_docs:
        chunks = text_splitter.split_text(doc["text"])
        if not chunks: continue
        
        embeddings = embedding_model.encode(chunks)
        ids = [f"{doc['doc_id']}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc["source"], "title": doc["title"]} for _ in chunks]
        
        collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
    
    print(f"--- Done! {len(new_docs)} documents added. ---")

if __name__ == "__main__":
    main()