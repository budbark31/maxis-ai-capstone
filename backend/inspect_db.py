import chromadb
import os

# --- Configuration ---
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
COLLECTION_NAME = "marywood_docs"

def main():
    """Queries the ChromaDB collection and prints the results."""
    # Connect to the database
    try:
        db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = db_client.get_collection(name=COLLECTION_NAME)
        print(f"Successfully connected to collection '{COLLECTION_NAME}'.")
        print(f"Total documents in collection: {collection.count()}")
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return

    # Loop to allow for multiple queries
    while True:
        query_text = input("\nEnter a query to inspect the database (or 'quit' to exit): ")
        if query_text.lower() == 'quit':
            break

        try:
            # Query the collection
            results = collection.query(
                query_texts=[query_text],
                n_results=3 # Ask for the top 3 results
            )

            # Print the results
            print("\n--- Top 3 Retrieved Documents ---")
            documents = results.get('documents', [[]])[0]
            if not documents:
                print("No documents found for this query.")
            else:
                for i, doc in enumerate(documents):
                    print(f"\n--- Document {i+1} ---")
                    # Print a snippet of the document for readability
                    snippet = ' '.join(doc.replace('\n', ' ').split())[:500]
                    print(f"{snippet}...")

        except Exception as e:
            print(f"An error occurred during query: {e}")

if __name__ == "__main__":
    main()