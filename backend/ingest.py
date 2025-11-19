import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
from io import BytesIO
import hashlib
import json
from tqdm import tqdm # Import tqdm for potential future use

# --- Configuration ---
load_dotenv()
START_URLS = [
    "https://www.marywood.edu/policy/handbooks/students/",
    "https://marywoodpacers.com/",
    "https://www.marywood.edu/about/"
]
ALLOWED_DOMAINS = ["marywood.edu", "marywoodpacers.com"]
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'visited_urls.json')
COLLECTION_NAME = "marywood_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
# Safe batch size for adding to ChromaDB
DB_ADD_BATCH_SIZE = 4000

os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper Functions ---
def get_url_hash(url: str) -> str:
    """Creates a short, safe MD5 hash from a URL."""
    return hashlib.md5(url.encode()).hexdigest()

def get_cached_content(url: str) -> str | None:
    """Checks for and returns cached content using a hashed filename."""
    filename = get_url_hash(url)
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [Cache HIT] Reading from cache: {url}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def cache_content(url: str, content: str):
    """Saves content to a local cache file using a hashed filename."""
    filename = get_url_hash(url)
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def scrape_pdf_content(url: str):
    """Downloads a PDF, extracts text, and returns the final URL and text."""
    try:
        response = requests.get(url, timeout=20, allow_redirects=True, headers=HEADERS)
        response.raise_for_status()
        final_url = response.url # Use the URL after redirects

        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            print(f"  [Warning] URL {final_url} is not a PDF ({content_type}). Skipping.")
            return final_url, ""

        with BytesIO(response.content) as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text + "\n" # Add newline between pages
        return final_url, text.strip()
    except Exception as e:
        print(f"  [Error] Failed to scrape PDF {url}: {e}")
        return url, "" # Return original URL on error

def load_visited_urls():
    """Loads the set of previously visited URLs from the state file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            print("[Warning] Could not read visited_urls.json. Starting fresh.")
            return set()
    return set()

def save_visited_urls(visited_urls):
    """Saves the current set of visited URLs to the state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(list(visited_urls), f, indent=2) # Add indent for readability
    except Exception as e:
        print(f"[Error] Could not save visited URLs: {e}")

def scrape_and_crawl(start_urls: list[str]):
    """Crawls websites, extracts content, handles caching and resumability."""
    print(f"--- Starting Crawl from {len(start_urls)} entry points ---")

    visited_urls = load_visited_urls()
    print(f"Loaded {len(visited_urls)} previously visited URLs.")

    # Start with URLs not yet visited
    urls_to_visit = set(start_urls) - visited_urls
    newly_processed_docs = [] # Collect docs processed in this session

    while urls_to_visit:
        current_url = urls_to_visit.pop()
        if current_url in visited_urls:
            continue

        print(f"Processing: {current_url}")
        visited_urls.add(current_url)

        final_url = current_url
        text = ""
        title = "Untitled"
        doc_id = get_url_hash(final_url) # Use hash as a stable doc_id
        soup = None

        # --- Caching Logic ---
        cached_content = get_cached_content(final_url)
        is_pdf_link = final_url.lower().endswith('.pdf')

        if cached_content:
            text = cached_content
            # Try to get title from soup even for cached HTML (if it's not a PDF)
            if not is_pdf_link:
                 try:
                    soup = BeautifulSoup(cached_content, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
                 except Exception:
                    title = "Cached HTML" # Fallback title
            else:
                 title = os.path.basename(final_url)

            # Add to list for potential DB update (will check later)
            newly_processed_docs.append({"text": text, "source": final_url, "title": title, "doc_id": doc_id})

            # Even if cached, parse HTML for new links (unless it's a PDF)
            if soup:
                 for link in soup.find_all('a', href=True):
                    absolute_url = urljoin(current_url, link['href']).split('#')[0] # Remove fragments
                    if any(domain in absolute_url for domain in ALLOWED_DOMAINS) and absolute_url not in visited_urls:
                        urls_to_visit.add(absolute_url)
            continue # Skip live fetch if cached

        # --- Live Fetch Logic ---
        print(f"  [Cache MISS] Scraping live: {current_url}")
        try:
            if is_pdf_link:
                final_url, text = scrape_pdf_content(current_url)
                title = os.path.basename(final_url)
                doc_id = get_url_hash(final_url)
                if text:
                    cache_content(final_url, text)
                    newly_processed_docs.append({"text": text, "source": final_url, "title": title, "doc_id": doc_id})
                continue # PDFs don't have links to follow

            else: # It's an HTML page
                response = requests.get(current_url, timeout=10, allow_redirects=True, headers=HEADERS)
                response.raise_for_status()
                final_url = response.url # Use final URL after redirects
                doc_id = get_url_hash(final_url)

                # Check if this final URL was already visited (due to redirect)
                if final_url in visited_urls:
                    print(f"  Redirected to already visited URL: {final_url}. Skipping.")
                    continue
                
                visited_urls.add(final_url) # Add final URL to visited set
                cache_content(final_url, response.text) # Cache the HTML content
                soup = BeautifulSoup(response.content, "html.parser")

                title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
                main_content = soup.find('main') or soup.body # Fallback to body

                if main_content:
                    text = main_content.get_text(separator="\n", strip=True)
                    newly_processed_docs.append({"text": text, "source": final_url, "title": title, "doc_id": doc_id})

                # Find new links from the live page
                if soup:
                     for link in soup.find_all('a', href=True):
                        absolute_url = urljoin(final_url, link['href']).split('#')[0] # Remove fragments
                        if any(domain in absolute_url for domain in ALLOWED_DOMAINS) and absolute_url not in visited_urls:
                            urls_to_visit.add(absolute_url)

        except requests.RequestException as e:
            print(f"  [Error] Could not fetch {current_url}: {e}")
        except Exception as e:
             print(f"  [Error] Could not process {current_url}: {e}")

        # Save progress periodically
        if len(visited_urls) % 20 == 0: # Save more frequently
            save_visited_urls(visited_urls)
            print(f"--- Progress Saved: {len(visited_urls)} URLs recorded ---")

    save_visited_urls(visited_urls) # Final save
    print(f"--- Crawl Complete. Processed {len(newly_processed_docs)} new/updated documents in this session. ---")
    return newly_processed_docs

def main():
    """Main function to run the incremental ingestion pipeline."""
    print("--- Starting Incremental Document Ingestion Pipeline ---")

    # --- Initialize DB ---
    print(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = db_client.get_or_create_collection(
             name=COLLECTION_NAME,
             # Specify the embedding function during creation/retrieval
             embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        )
        print(f"Connected to collection '{COLLECTION_NAME}'. It has {collection.count()} documents.")
    except Exception as e:
         print(f"[Error] Could not connect to Chroma DB: {e}. Aborting.")
         return

    # --- Get existing doc IDs (more efficient way) ---
    try:
        existing_docs_result = collection.get(include=[]) # Only need IDs
        existing_doc_ids = set(existing_docs_result['ids'])
        print(f"Found {len(existing_doc_ids)} existing document chunks in the database.")
    except Exception as e:
        print(f"[Warning] Could not retrieve existing IDs from DB: {e}. May re-add duplicates.")
        existing_doc_ids = set()

    # --- Crawl for new/updated documents ---
    docs_to_process = scrape_and_crawl(START_URLS)
    if not docs_to_process:
        print("No new documents found during crawl. Ingestion complete.")
        return

    # --- Process and add new documents ---
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model loaded.")
    except Exception as e:
        print(f"[Error] Could not load embedding model: {e}. Aborting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    new_chunks_count = 0

    # Process documents individually to handle potential errors
    for doc in tqdm(docs_to_process, desc="Processing Documents"):
        doc_id = doc["doc_id"]
        chunks_for_doc = []
        metadatas_for_doc = []
        ids_for_doc = []

        try:
            raw_chunks = text_splitter.split_text(doc["text"])
            for i, chunk in enumerate(raw_chunks):
                chunk_id = f"{doc_id}_{i}"
                # Check if this specific chunk ID already exists
                if chunk_id not in existing_doc_ids:
                    chunks_for_doc.append(chunk)
                    metadatas_for_doc.append({"source": doc["source"], "title": doc["title"], "doc_id": doc_id})
                    ids_for_doc.append(chunk_id)

            if not chunks_for_doc:
                # print(f"  - Document {doc_id} already fully processed.")
                continue # Skip if no new chunks for this doc

            # Create embeddings for the new chunks of this document
            embeddings = embedding_model.encode(chunks_for_doc, show_progress_bar=False) # Progress bar inside loop is too noisy

            # Add the new chunks for this document in smaller batches
            for j in range(0, len(chunks_for_doc), DB_ADD_BATCH_SIZE):
                 collection.add(
                     embeddings=embeddings[j:j+DB_ADD_BATCH_SIZE],
                     documents=chunks_for_doc[j:j+DB_ADD_BATCH_SIZE],
                     metadatas=metadatas_for_doc[j:j+DB_ADD_BATCH_SIZE],
                     ids=ids_for_doc[j:j+DB_ADD_BATCH_SIZE]
                 )
            new_chunks_count += len(chunks_for_doc)
            print(f"  - Added {len(chunks_for_doc)} new chunks for document {doc_id}.")

        except Exception as e:
            print(f"  [Error] Failed to process or add document {doc_id} ({doc.get('source')}): {e}")

    print(f"\nSuccessfully added {new_chunks_count} new document chunks.")
    print(f"Collection now has {collection.count()} total documents.")
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()