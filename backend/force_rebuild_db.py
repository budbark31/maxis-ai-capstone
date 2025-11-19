import os
import chromadb
import logging
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# --- Configuration ---
load_dotenv()

# Paths and constants
BASE_DIR = os.path.dirname(__file__)
CHROMA_DB_PATH = os.path.join(BASE_DIR, 'chroma_db')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
STATE_FILE = os.path.join(BASE_DIR, 'visited_urls.json')
PROGRESS_FILE = os.path.join(BASE_DIR, 'rebuild_progress.json')
LOG_FILE = os.path.join(BASE_DIR, 'rebuild.log')

# Tunables (env overrides supported)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "marywood_docs")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # 'cuda' if GPU is available
CHUNK_SIZE = int(os.getenv("REBUILD_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("REBUILD_CHUNK_OVERLAP", "300"))
PROCESS_BATCH_SIZE = int(os.getenv("REBUILD_PROCESS_BATCH_SIZE", "500"))
DB_ADD_BATCH_SIZE = int(os.getenv("REBUILD_DB_ADD_BATCH_SIZE", "4000"))
MAX_FILES = int(os.getenv("REBUILD_MAX_FILES", "0"))  # 0 = no limit
RESUME = os.getenv("REBUILD_RESUME", "true").lower() not in ("0", "false", "no")
MAX_WORKERS = int(os.getenv("REBUILD_MAX_WORKERS", str(min(8, (os.cpu_count() or 4)))))
MIN_CHUNK_LEN = int(os.getenv("REBUILD_MIN_CHUNK_LEN", "80"))

def setup_logging() -> None:
    """Configure logging to both console and file with timestamps."""
    fmt = '[%(asctime)s] %(levelname)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding='utf-8')
        ]
    )

def safe_get_url(hash_to_url: Dict[str, str], filename: str) -> str:
    """Return mapped source URL for a cache filename if known, else a fallback tag."""
    return hash_to_url.get(filename, "cached_document")

def process_file(filename: str, text_splitter: RecursiveCharacterTextSplitter, hash_to_url: Dict[str, str]) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    """Read and chunk one cached file, returning chunks, metadatas, and ids.

    Skips very short chunks. De-duplicates chunks within the file.
    """
    filepath = os.path.join(CACHE_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = text_splitter.split_text(text)
        seen: Set[str] = set()
        out_chunks: List[str] = []
        out_metas: List[Dict[str, str]] = []
        out_ids: List[str] = []
        for j, chunk in enumerate(chunks):
            if len(chunk.strip()) < MIN_CHUNK_LEN:
                continue
            h = hashlib.md5(chunk.strip().encode('utf-8')).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            chunk_id = f"{filename}_{j}"
            out_chunks.append(chunk)
            out_metas.append({"source": safe_get_url(hash_to_url, filename)})
            out_ids.append(chunk_id)
        return out_chunks, out_metas, out_ids
    except Exception as e:
        logging.warning(f"Could not read or process file {filename}: {e}")
        return [], [], []

def main():
    """
    Forces a rebuild of the ChromaDB database from cache, using batches for
    both file processing and database additions to handle massive datasets.
    """
    setup_logging()
    logging.info("--- Starting Enterprise Database Rebuild from Cache ---")

    if not os.path.exists(CACHE_DIR):
        logging.error(f"Cache directory not found at: {CACHE_DIR}")
        return

    cached_files = os.listdir(CACHE_DIR)
    logging.info(f"Found {len(cached_files)} files in the cache.")

    # Load progress (already processed filenames)
    processed_files: Set[str] = set()
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as pf:
                data = json.load(pf)
                if isinstance(data, list):
                    processed_files = set(data)
            logging.info(f"Loaded progress file with {len(processed_files)} processed files.")
    except Exception as e:
        logging.warning(f"Could not read progress file: {e}")

    # Build a mapping from cache filename (md5 hash) to original URL if possible
    hash_to_url: Dict[str, str] = {}
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                visited_urls = json.load(f)
                for url in visited_urls:
                    h = hashlib.md5(url.encode()).hexdigest()
                    hash_to_url[h] = url
            logging.info(f"Reconstructed URL mapping for {len(hash_to_url)} cached files from visited_urls.json.")
        else:
            logging.warning("visited_urls.json not found. 'source' metadata will fall back to 'cached_document'.")
    except Exception as e:
        logging.warning(f"Could not read visited_urls.json for URL mapping: {e}")

    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device {EMBEDDING_DEVICE} ...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    logging.info("Model loaded.")

    logging.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    existing_names = [c.name for c in db_client.list_collections()]
    if COLLECTION_NAME in existing_names and not RESUME:
        logging.info(f"Deleting existing collection '{COLLECTION_NAME}' (RESUME=False)...")
        db_client.delete_collection(name=COLLECTION_NAME)

    # get_or_create supports resume without wiping data
    collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    )
    try:
        logging.info(f"Connected to collection. Current count: {collection.count()} documents.")
    except Exception:
        logging.info("Connected to collection.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Apply MAX_FILES limit and skip already processed
    if MAX_FILES > 0:
        to_process_all = [f for f in cached_files if f not in processed_files][:MAX_FILES]
    else:
        to_process_all = [f for f in cached_files if f not in processed_files]

    total_chunks_added = 0
    for i in range(0, len(to_process_all), PROCESS_BATCH_SIZE):
        batch_files = to_process_all[i:i+PROCESS_BATCH_SIZE]
        logging.info(f"\n--- Processing file batch {i//PROCESS_BATCH_SIZE + 1}/{(len(to_process_all) + PROCESS_BATCH_SIZE - 1)//PROCESS_BATCH_SIZE} ---")

        all_chunks: List[str] = []
        all_metadatas: List[Dict[str, str]] = []
        all_ids: List[str] = []

        logging.info("  - Reading and chunking files in parallel...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_file, filename, text_splitter, hash_to_url): filename for filename in batch_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="  Chunking Files"):
                filename = futures[future]
                chunks, metas, ids = future.result()
                if chunks:
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metas)
                    all_ids.extend(ids)
                    processed_files.add(filename)

        if not all_chunks:
            logging.info("  - No chunks created from this batch, skipping.")
            continue
        
        logging.info(f"  - Created {len(all_chunks)} chunks. Now creating embeddings...")
        embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)

        # --- KEY UPGRADE: Add to DB in smaller batches ---
        logging.info(f"  - Adding {len(all_chunks)} chunks to the database in smaller batches...")
        for j in tqdm(range(0, len(all_chunks), DB_ADD_BATCH_SIZE), desc="  Adding to DB"):
            emb_batch = embeddings[j:j+DB_ADD_BATCH_SIZE]
            doc_batch = all_chunks[j:j+DB_ADD_BATCH_SIZE]
            meta_batch = all_metadatas[j:j+DB_ADD_BATCH_SIZE]
            id_batch = all_ids[j:j+DB_ADD_BATCH_SIZE]

            # Try add; if it fails due to duplicate IDs, try upsert or add only missing
            try:
                collection.add(
                    embeddings=emb_batch,
                    documents=doc_batch,
                    metadatas=meta_batch,
                    ids=id_batch
                )
            except Exception as e:
                logging.warning(f"Add failed for a batch of {len(id_batch)} (will attempt upsert/missing-only): {e}")
                # Try upsert if available
                if hasattr(collection, 'upsert'):
                    try:
                        collection.upsert(
                            embeddings=emb_batch,
                            documents=doc_batch,
                            metadatas=meta_batch,
                            ids=id_batch
                        )
                        continue
                    except Exception as e2:
                        logging.warning(f"Upsert also failed; will try missing-only add: {e2}")
                # Fallback: get existing subset and add only missing
                try:
                    existing = set()
                    try:
                        res = collection.get(ids=id_batch)
                        if res and 'ids' in res and res['ids']:
                            for row in res['ids']:
                                if isinstance(row, list):
                                    existing.update(row)
                                else:
                                    existing.add(row)
                    except Exception:
                        # If get(ids=...) not supported, skip filtering
                        pass
                    if existing:
                        missing_idx = [k for k, _id in enumerate(id_batch) if _id not in existing]
                        if missing_idx:
                            collection.add(
                                embeddings=[emb_batch[k] for k in missing_idx],
                                documents=[doc_batch[k] for k in missing_idx],
                                metadatas=[meta_batch[k] for k in missing_idx],
                                ids=[id_batch[k] for k in missing_idx]
                            )
                except Exception as e3:
                    logging.error(f"Failed to add remaining missing IDs in batch: {e3}")
        # --- END OF UPGRADE ---

        total_chunks_added += len(all_chunks)
        current_count = 0
        try:
            current_count = collection.count()
        except Exception:
            pass
        logging.info(f"  - Batch added. Total chunks in DB: {current_count}")

        # Persist progress after each batch
        try:
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as pf:
                json.dump(sorted(list(processed_files)), pf)
        except Exception as e:
            logging.warning(f"Could not write progress file: {e}")

    logging.info("\n--- Database Rebuild Complete ---")
    logging.info(f"Successfully processed files: {len(processed_files)}")
    logging.info(f"Successfully added a total of {total_chunks_added} chunks (pre-dedup within batch).")
    try:
        logging.info(f"Final collection count: {collection.count()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

