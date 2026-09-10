import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Initializing HuggingFace embeddings model...", flush=True)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
    return _embeddings

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store...", flush=True)

    # Ensure clean directory/collection for fresh session
    if not transcript or not transcript.strip():
        docs = [Document(page_content="No transcript content available for this session.", metadata={"chunk_index": 0})]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(transcript.strip())
        if not chunks:
            chunks = ["No speech content detected."]

        docs = [
            Document(page_content=chunk, metadata={'chunk_index': i})
            for i, chunk in enumerate(chunks)
        ]

    embeddings = get_embeddings()

    # Reset collection if exists to avoid mixing previous sessions
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )



