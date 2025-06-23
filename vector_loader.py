import os
print("Current Working Directory:", os.getcwd())
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load your AASHTO soil knowledge base
AASHTO_KB = """... (content truncated for brevity - full content from above will be restored here) ..."""

# Prepare documents
sections = AASHTO_KB.split("### ")
docs = [Document(page_content=f"### {section.strip()}") for section in sections if section.strip()]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and persist vector store
persist_dir = "aashto_vectorstore"
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_dir)
vectorstore.persist()
print("✅ Vector store built and saved to 'aashto_vectorstore/'")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)
print("✅ AASHTO vector store is loaded.")
print("📁 Collection Name:", vectorstore._collection.name)
print("📦 Stored Docs Count:", vectorstore._collection.count())

