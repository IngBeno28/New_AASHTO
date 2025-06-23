from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

# Load model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"  # You can change this to something lighter if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create LLM pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="aashto_vectorstore", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Build the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",  # You can try 'stuff' or 'map_reduce' too
    retriever=retriever,
    return_source_documents=True
)

def ask_specbot(query):
    try:
        result = qa_chain(query)
        answer = result["result"]
        source_docs = result["source_documents"]
        source_info = "\n\n📚 **Source Excerpts:**\n" + "\n".join([
            f"- _{doc.page_content.strip()[:250]}..._" for doc in source_docs
        ])
        return answer, source_info, source_docs
    except Exception as e:
        st.warning("LLM response unavailable. Returning default soil classification description.")
        return (
            "This soil falls under a known AASHTO group, but model insights could not be retrieved.",
            "",
            []
        )
