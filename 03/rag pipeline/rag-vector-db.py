# 03/rag pipeline/rag-vector-db.py
# ------------------------------------ Imports ----------------------------------
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET


# ------------------------------------ Variables / Constants ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# AI Model
MODEL = "gpt-4o-mini"


# ------------------------------------ Configure API Keys / Tokens ----------------------------------
# Path to the .env file
env_path = os.path.join(parent_dir, ".env")

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API keys stored in the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")  # https://openai.com/api/


print(openai_api_key[:5])


    
# ------------------------------------ Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualization of what's going on
    """
    banner_len = len(text)
    mid = 49 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 50)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 50)


def get_loader_for_path(path: Path):
    """
    Return an appropriate LangChain loader based on file extension.
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")

    if suffix == ".pdf":
        try:
            return PyPDFLoader(str(path))  # fast, but limited
        except:
            return UnstructuredPDFLoader(str(path))  # fallback, more powerful

    if suffix == ".docx":
        return Docx2txtLoader(str(path))

    if suffix in {".html", ".htm"}:
        return UnstructuredHTMLLoader(str(path))

    # Fallback: try UnstructuredFileLoader for anything else
    return UnstructuredFileLoader(str(path))


# ------------------------------------ Read in Documents using LangChain's Loader ----------------------------------
print_banner("Read in Documents using LangChain's Loader")

print("\nLoading documents back in with metadata")
all_docs = []

knowledge_base_dir = Path(script_dir) / "knowledge base"

# Walk the knowledge base directory and load all supported files
for path in knowledge_base_dir.rglob("*"):
    if not path.is_file():
        continue

    try:
        loader = get_loader_for_path(path)
    except Exception as e:
        print(f"⚠️ Skipping {path} (no loader): {e}")
        continue

    try:
        loaded_docs = loader.load()
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        continue

    for d in loaded_docs:
        # Common metadata
        d.metadata["source_path"] = str(path)
        d.metadata["source_name"] = path.name


        all_docs.append(d)
        print(f"✅ Loaded: {path}")

# # Preview the first document
# if all_docs:
#     print("\nExample document at index 0:")
#     print(all_docs[0])
# else:
#     print("\nNo documents found in knowledge base.")


# ------------------------------------ Break Down Documents into Overlapping Chunks ----------------------------------
print_banner("Break Down Documents into Overlapping Chunks")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(all_docs)


# ------------------------------------ Inspect the Chunks Created by LangChain ----------------------------------
print_banner("Inspect the Chunks Created by LangChain")

print(f"*{len(chunks)} number of chunks were created")

# # Inspect chunks:
# for chunk in chunks:
#     print(chunk)
#     print('\n')


# ------------------------------------ Vector Embeddings ----------------------------------
print_banner("Vector Embeddings")
# Create an embedding model using OpenAI's embedding API
# The langchain-openai library (specifically OpenAIEmbeddings and ChatOpenAI) automatically looks for the 
# 'OPENAI_API_KEY' environment variable. When you instantiate OpenAIEmbeddings():
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Define the path where the vector database will be stored
faiss_vector_store = script_dir + "\\vectorstore_db"

# Create a FAISS (Facebook AI Similarity Search) vector store from the pre-chunked documents.
# This will generate vector embeddings and store them in memory.  
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# Analyze the vectorstore
total_vectors = vectorstore.index.ntotal
dimensions = vectorstore.index.d

print(f"*There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")

# Save FAISS index + metadata to disk
vectorstore.save_local(faiss_vector_store)
print(f"✅ Saved FAISS index to: {faiss_vector_store}")

# To load vector db in
# print("Load Vector Store Back In")
# vectorstore = FAISS.load_local(
#     faiss_vector_store,
#     embeddings,
#     allow_dangerous_deserialization=True,  # required in newer langchain versions
# )


# ------------------------------------ 3D Embedding Visualization ----------------------------------
print_banner("3D Visualization of Embeddings")

# 1. Compute embeddings for each text chunk
texts = [c.page_content for c in chunks]
X = embeddings.embed_documents(texts)
X = np.array(X)  # shape: (num_chunks, 1536)

print(f"Embedding matrix: {X.shape}")

# 2. Reduce dimensions to 3D
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X)

# 3. Plot using matplotlib's 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=40)

# Label some points with PMID or short identifiers
for i, doc in enumerate(chunks):
    pmid = doc.metadata.get("pmid", "")
    if i % 5 == 0:  # label every 5th to avoid clutter
        ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], pmid)

ax.set_title("3D PCA Visualization of Chunk Embeddings")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.tight_layout()
plt.show()


# # --------------------------- 4D Visualization (Color Encoded) ---------------------------
# print_banner("4D PCA Visualization (3D Scatter + Color)")

# texts = [c.page_content for c in chunks]
# X = np.array(embeddings.embed_documents(texts))

# # Reduce to 4D
# pca = PCA(n_components=4)
# X_4d = pca.fit_transform(X)

# # First 3 dimensions → axes
# x, y, z = X_4d[:, 0], X_4d[:, 1], X_4d[:, 2]
# # Fourth dimension → color scale
# c = X_4d[:, 3]

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# scatter = ax.scatter(x, y, z, c=c, cmap="viridis", s=50)

# # Add color bar to show 4th dimension
# cbar = plt.colorbar(scatter, label="4th PCA Dimension Value")

# ax.set_title("PubMed Embedding Visualization (4D Encoded in Color)")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")

# plt.show()


# ------------------------------------ Testing the LLM Integration with RAG ----------------------------------
print_banner("Testing the LLM Integration with RAG")

# Create a new Chat with OpenAI
llm = ChatOpenAI(model=MODEL, temperature=0.3)

# The retriever's role is to fetch relevant chunks of information from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

system_prompt = """
You are a helpful assistant that answers questions about LLM security pratices
Use ONLY the following context to answer. If the answer is not in the context, say you don't know.
"""

# Prompt template: how the LLM should use retrieved context
prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        system_prompt
    ),
    (
        "human",
        "Question: {input}\n\nContext:\n{context}"
    ),
])

# Chain that stuffs retrieved docs into the prompt
qa_chain = create_stuff_documents_chain(llm, prompt)

# Full RAG chain: retrieval + question answering
rag_chain = create_retrieval_chain(retriever, qa_chain)

query = "What genus does the monkeypox virus belong to?"
# Orthopoxvirus
result = rag_chain.invoke({"input": query})


print("\n --- RAG Question ---")
print(system_prompt)
print(f"Question\n\n{query}")

print("\n--- RAG Answer ---")
print(result["answer"])

print("\n--- Sources ---")
for doc in result["context"]:
    print(f"Source: {doc.metadata.get('source_name')}")