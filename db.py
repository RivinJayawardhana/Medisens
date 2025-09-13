# create_vector_db_from_pdfs.py

import os
import pickle
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdfs(pdf_paths: List[str]):
    """
    Load text documents from given PDF file paths.
    """
    documents = []
    for file_path in pdf_paths:
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"✅ Loaded {file_path}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def create_vector_db(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create a FAISS vector store from documents using HuggingFace embeddings.
    """
    print("📐 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(docs, embeddings)


def save_vector_db(vectorstore, output_path="faiss_store_pdfs.pkl"):
    """
    Save the FAISS vector store to a pickle file.
    """
    print(f"💾 Saving vector DB to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print("✅ Vector DB saved successfully!")


def load_vector_db(input_path="faiss_store_pdfs.pkl"):
    """
    Load the FAISS vector store from a pickle file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Vector DB file {input_path} not found.")
    with open(input_path, "rb") as f:
        vectorstore = pickle.load(f)
    print("📂 Vector DB loaded successfully!")
    return vectorstore


def create_vector_db_from_pdfs(pdf_paths, output_path="faiss_store_pdfs.pkl"):
    """
    Full pipeline: load PDFs → split → embed → save vector DB.
    """
    print("📄 Loading PDF documents...")
    pdf_docs = load_pdfs(pdf_paths)

    if not pdf_docs:
        raise ValueError("No documents were loaded from the provided PDFs.")

    print("✂️ Splitting documents...")
    docs = split_documents(pdf_docs)

    vectorstore = create_vector_db(docs)

    save_vector_db(vectorstore, output_path)
    return vectorstore


# Example usage
if __name__ == "__main__":
    # Replace with your local PDF file paths
    pdf_files = ["./sample.pdf"]
    vectorstore = create_vector_db_from_pdfs(pdf_files)

   
