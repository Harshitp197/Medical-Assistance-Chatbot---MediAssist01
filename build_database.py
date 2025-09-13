import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import time

print("Starting the database build process...")

# --- 1. LOAD AND PREPROCESS THE DATA ---
try:
    print("Loading the dataset...")
    # This now uses a relative path, which is portable and works with Docker.
    df = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
    
    df['text'] = df['Question'] + " " + df['Answer']
    print("Combined 'Question' and 'Answer' columns.")

except FileNotFoundError:
    print("Error: train.csv not found. Please ensure it is in the same directory as this script.")
    exit()

# --- 2. INITIALIZE THE EMBEDDING MODEL ---
print("Initializing the embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

# --- 3. SETUP CHROMA DATABASE ---
client = chromadb.PersistentClient(path="db")
collection_name = "medical_qa"
print(f"Setting up ChromaDB collection: '{collection_name}'...")
collection = client.get_or_create_collection(name=collection_name)
print("Collection is ready.")

# --- 4. EMBED AND STORE THE DATA ---
batch_size = 256 # Increased for potentially better performance
total_docs = len(df)
print(f"Preparing to process {total_docs} documents...")

start_time = time.time()

for i in range(0, total_docs, batch_size):
    batch_df = df.iloc[i:i+batch_size]
    
    documents = batch_df['text'].tolist()
    ids = [str(idx) for idx in batch_df.index]
    
    embeddings = model.encode(documents, show_progress_bar=False).tolist()
    
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids
    )
    
    print(f"Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}...")

end_time = time.time()
print("\n--- Database Build Complete ---")
print(f"Total documents processed: {total_docs}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

count = collection.count()
print(f"The collection '{collection_name}' now contains {count} items.")
print("The database is saved in the 'db' folder.")