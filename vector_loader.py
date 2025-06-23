print("🚀 Script started...")
import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

print("📁 Currently running:", os.path.abspath(__file__))
print("📁 Current Working Directory:", os.getcwd())

# Load your AASHTO soil knowledge base
AASHTO_KB = \"\"\"### A-1-a
- **Type**: Well-graded gravel and sand with little to no fines
- **Plasticity Index**: ≤ 6
- **Liquid Limit**: ≤ 40
- **Typical Use**: Ideal for subbase and base courses
- **Strength**: High load-bearing capacity
- **Limitations**: Minimal; excellent performance

### A-1-b
- **Type**: Similar to A-1-a but more coarse (more gravel)
- **Plasticity Index**: ≤ 6
- **Liquid Limit**: ≤ 40
- **Typical Use**: Subbase and base for highways
- **Strength**: Excellent
- **Limitations**: Rarely problematic

### A-2-4
- **Type**: Silty or clayey sand
- **Plasticity Index**: ≤ 10
- **Liquid Limit**: < 40
- **Typical Use**: Embankments, subgrades under light traffic
- **Strength**: Fair to good if compacted
- **Limitations**: Loses strength when wet

### A-2-5
- **Type**: Clayey sand with moderate fines
- **Plasticity Index**: > 10
- **Liquid Limit**: < 40
- **Typical Use**: Subgrades for lightly loaded roads
- **Strength**: Moderate
- **Limitations**: Moisture sensitivity

### A-2-6
- **Type**: Silty or clayey sand with higher PI
- **Plasticity Index**: > 10
- **Liquid Limit**: > 40
- **Typical Use**: Embankments, some subgrades
- **Strength**: Moderate
- **Limitations**: Expansive in wet conditions

### A-2-7
- **Type**: Very silty or clayey sand
- **Plasticity Index**: > 10
- **Liquid Limit**: > 40
- **Typical Use**: Low-volume roads
- **Strength**: Marginal
- **Limitations**: High expansion and poor drainage

### A-3
- **Type**: Clean sand
- **Plasticity Index**: Non-plastic
- **Liquid Limit**: Not applicable
- **Typical Use**: Subbase for roads and runways
- **Strength**: Good drainage, low cohesion
- **Limitations**: Needs containment under heavy loads

### A-4
- **Type**: Silty soils
- **Plasticity Index**: ≤ 10
- **Liquid Limit**: ≤ 40
- **Typical Use**: General fill or subgrade
- **Strength**: Fair when compacted
- **Limitations**: Poor under moisture variation

### A-5
- **Type**: Silty soils with higher LL
- **Plasticity Index**: ≤ 10
- **Liquid Limit**: > 40
- **Typical Use**: Low-volume roads, landscaping fill
- **Strength**: Low
- **Limitations**: Sensitive to water, frost-susceptible

### A-6
- **Type**: Clayey soil
- **Plasticity Index**: > 10
- **Liquid Limit**: ≤ 40
- **Typical Use**: Subgrade material
- **Strength**: Moderate
- **Limitations**: Shrink-swell potential

### A-7-5
- **Type**: Silty clay
- **Plasticity Index**: > 10
- **Liquid Limit**: > 40
- **Typical Use**: Very low-traffic subgrades
- **Strength**: Weak
- **Limitations**: Moisture sensitive, poor drainage

### A-7-6
- **Type**: Plastic clay
- **Plasticity Index**: Very high
- **Liquid Limit**: > 40
- **Typical Use**: Not recommended for structural support
- **Strength**: Very low
- **Limitations**: Severe expansion, shrinkage, poor strength
\"\"\"


# Final bulletproof group-by-group extraction
groups = [
    "A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", "A-2-7",
    "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"
]

docs = []
for group in groups:
    pattern = rf"### {group}[\\s\\S]*?(?=### A-|\\Z)"
    match = re.search(pattern, AASHTO_KB)
    if match:
        docs.append(Document(page_content=match.group().strip()))
    else:
        print(f"❌ Failed to extract: {group}")
        print("\\n✅ Extracted groups:")
    for i, doc in enumerate(docs):
    print(f"{i+1}. {doc.page_content.splitlines()[0]}")


print("📄 Number of document chunks created:", len(docs))
print("🧪 Preview of first heading:\\n", docs[0].page_content[:100])

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
persist_dir = "aashto_vectorstore"
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=persist_dir)

# Debug output
print("✅ AASHTO vector store is loaded.")
print("📁 Collection Name:", vectorstore._collection.name)
print("📦 Number of stored documents:", vectorstore._collection.count())
print("✅ Vector store fully prepared.")
"""

final_path = "/mnt/data/vector_loader_final_fixed.py"
with open(final_path, "w") as f:
    f.write(final_vector_loader.strip())

final_path
