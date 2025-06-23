print("🚀 Script started...")
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

print("📁 Current Working Directory:", os.getcwd())

# Load your AASHTO soil knowledge base
AASHTO_KB = """... (your full soil descriptions go here) ..."""

# Prepare documents
sections = AASHTO_KB.split("### ")
docs = [Document(page_content=f"### {section.strip()}") for section in sections if section.strip()]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store (no need to call .persist() explicitly in Chroma ≥0.4.0)
persist_dir = "aashto_vectorstore"
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_dir)

# Debug: Show results
print("✅ AASHTO vector store is loaded.")
print("📁 Collection Name:", vectorstore._collection.name)
print("📦 Number of stored documents:", vectorstore._collection.count())
