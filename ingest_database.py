# ingest_database.py
import os
import re
import pickle
from typing import List, Tuple
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import faiss
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_PATH = "data"
INDEX_PATH = "faiss_index/faiss.index"

def parse_pdf_from_path(path: str) -> Tuple[str, str]:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
                text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
                text = re.sub(r"\n\s*\n", "\n\n", text)
                full_text += "\n" + text
        filename = os.path.basename(path)
        return full_text.strip(), filename

def text_to_docs(text: str, filename: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,        # ✨ Changed from 3000 ➔ 1200
        chunk_overlap=100,      # ✨ Changed from 300 ➔ 100
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"filename": filename, "chunk": i}) for i, chunk in enumerate(chunks)]

def build_faiss_index(documents: List[Document], openai_api_key: str, index_path=INDEX_PATH):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    index = FAISS.from_documents(documents, embeddings)
    faiss.write_index(index.index, index_path)
    with open(f"{index_path}.pkl", "wb") as f:
        pickle.dump({
            "docstore": index.docstore,
            "index_to_docstore_id": index.index_to_docstore_id,
        }, f)
    return index

def load_faiss_index(openai_api_key: str, index_path=INDEX_PATH):
    if not os.path.exists(index_path) or not os.path.exists(f"{index_path}.pkl"):
        raise FileNotFoundError("Index or metadata not found.")
    index = faiss.read_index(index_path)
    with open(f"{index_path}.pkl", "rb") as f:
        meta = pickle.load(f)
    return FAISS(
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        index=index,
        docstore=meta["docstore"],
        index_to_docstore_id=meta["index_to_docstore_id"]
    )

def process_directory_to_faiss(data_path: str, openai_api_key: str):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            text, fname = parse_pdf_from_path(os.path.join(data_path, filename))
            docs = text_to_docs(text, fname)
            documents.extend(docs)
    return build_faiss_index(documents, openai_api_key)

if __name__ == "__main__":
    process_directory_to_faiss(DATA_PATH, OPENAI_API_KEY)
