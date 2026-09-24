"""
vector_db.py

ClinicaRAG - Member 1

Responsibilities
----------------
1. Load complete MedQuAD dataset
2. Prepare medical documents
3. Chunk long answers
4. Generate embeddings
5. Build ChromaDB vector database
6. Semantic search over medical knowledge
"""

from pathlib import Path
from typing import List, Dict
import logging

import pandas as pd

import chromadb

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "medquad_complete"
    / "medquad_complete.csv"
)

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
)

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# Dataset Loader
# ==========================================================

def load_medquad() -> pd.DataFrame:
    """
    Loads the complete MedQuAD dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    logger.info("Loading MedQuAD dataset...")

    df = pd.read_csv(DATASET_PATH)

    logger.info(f"Loaded {len(df)} records.")

    return df


# ==========================================================
# Dataset Cleaning
# ==========================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes missing values and duplicates.
    """

    logger.info("Cleaning dataset...")

    df = df.copy()

    df = df.dropna(subset=["Question", "Answer"])

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    logger.info(f"Remaining records: {len(df)}")

    return df


# ==========================================================
# Create Medical Documents
# ==========================================================

def create_documents(df: pd.DataFrame) -> List[Dict]:
    """
    Converts dataframe rows into document dictionaries.
    """

    logger.info("Creating medical documents...")

    documents = []

    for idx, row in df.iterrows():

        documents.append(
            {
                "id": str(idx),

                "question": row["Question"],

                "answer": row["Answer"],

                "qtype": row["qtype"],

                "text":
                    f"Question: {row['Question']}\n\n"
                    f"Answer: {row['Answer']}"
            }
        )

    logger.info(f"Created {len(documents)} documents.")

    return documents


# ==========================================================
# Text Chunking
# ==========================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    """
    Splits long text into overlapping chunks.
    """

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# ==========================================================
# Prepare Chunks
# ==========================================================

def prepare_chunks(
    documents: List[Dict]
) -> List[Dict]:
    """
    Creates chunks for every medical document.
    """

    logger.info("Creating chunks...")

    chunked_documents = []

    chunk_id = 0

    for doc in documents:

        pieces = chunk_text(doc["text"])

        for piece in pieces:

            chunked_documents.append(
                {
                    "id": str(chunk_id),

                    "question": doc["question"],

                    "answer": doc["answer"],

                    "qtype": doc["qtype"],

                    "text": piece,
                }
            )

            chunk_id += 1

    logger.info(
        f"Generated {len(chunked_documents)} chunks."
    )

    return chunked_documents

# ==========================================================
# Load Embedding Model
# ==========================================================

def load_embedding_model() -> SentenceTransformer:
    """
    Loads the SentenceTransformer model.
    """

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Embedding model loaded successfully.")

    return model


# ==========================================================
# Generate Embeddings
# ==========================================================

def generate_embeddings(
    model: SentenceTransformer,
    chunks: List[Dict],
    batch_size: int = 64,
):
    """
    Generates embeddings for all text chunks.
    """

    logger.info("Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    logger.info(f"Generated {len(embeddings)} embeddings.")

    return embeddings

# ==========================================================
# Create ChromaDB Client
# ==========================================================

def create_vector_database() -> chromadb.Collection:
    """
    Creates a persistent ChromaDB client.
    """

    logger.info("Creating ChromaDB client...")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    collection = client.get_or_create_collection(
        name="medical_knowledge",
        metadata={
            "description": "ClinicaRAG Medical Knowledge Base"
        }
    )

    logger.info("Collection ready.")

    return collection


# ==========================================================
# Store Embeddings
# ==========================================================

def store_embeddings(
    collection,
    chunks,
    embeddings,
    CHROMA_BATCH_SIZE = 500,
):
    """
    Stores embeddings into ChromaDB.
    """

    logger.info("Storing embeddings into ChromaDB...")

    total = len(chunks)

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        ids = [
            chunk["id"]
            for chunk in batch_chunks
        ]

        documents = [
            chunk["text"]
            for chunk in batch_chunks
        ]

        metadatas = [
            {
                "question": chunk["question"],
                "answer": chunk["answer"],
                "qtype": chunk["qtype"],
                "source": "MedQuAD",
            }
            for chunk in batch_chunks
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings.tolist(),
            metadatas=metadatas,
        )

    logger.info("All embeddings stored successfully.")

# ==========================================================
# Query Medical Knowledge Base
# ==========================================================

def query_medical_kb(
    collection,
    model: SentenceTransformer,
    query_text: str,
    top_k: int = 3,
):
    """
    Searches the medical knowledge base.

    Parameters
    ----------
    collection
        ChromaDB collection.

    model
        SentenceTransformer model.

    query_text
        User medical query.

    top_k
        Number of results.

    Returns
    -------
    dict
        ChromaDB query results.
    """

    logger.info(f"Searching: {query_text}")

    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    return results

def build_vector_database():
    """
    Builds the complete medical vector database.
    """

    df = load_medquad()

    df = clean_dataset(df)

    docs = create_documents(df)

    chunks = prepare_chunks(docs)

    DEVELOPMENT_MODE = True

    if DEVELOPMENT_MODE:
        chunks = chunks[:5000]

    model = load_embedding_model()

    embeddings = generate_embeddings(
        model,
        chunks,
    )

    collection = create_vector_database()

    store_embeddings(
        collection,
        chunks,
        embeddings,
    )

    return collection, model



# ==========================================================
# Main (Temporary)
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ClinicaRAG Vector Database")
    print("=" * 60)

    collection, model = build_vector_database()

    print("\n")
    print("=" * 60)
    print("Testing Medical Search")
    print("=" * 60)

    query = "What causes diabetes?"

    results = query_medical_kb(
        collection,
        model,
        query,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print(f"\nQuery: {query}\n")

    for i, (doc, meta) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):

        print("-" * 60)

        print(f"Result {i}")

        print(f"Question : {meta['question']}")

        print(f"Type     : {meta['qtype']}")

        print()

        print(doc[:500])

        print()
