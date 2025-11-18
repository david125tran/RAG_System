# ------------------------------------ Imports ----------------------------------
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import requests
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET


# ------------------------------------ Constants / Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# What to search pubmed for
PUBMED_SEARCH = "plasma metanephrine"

# Create a directory for files we download
output_dir = Path(script_dir) / "pubmed_abstracts" / PUBMED_SEARCH
output_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------ Configure API Keys / Tokens ----------------------------------
# Specify the path to the .env file
env_path = script_dir + "\\.env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

# Access the API keys stored in the environment variable
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')            # https://openai.com/api/
pubmed_api_key = os.getenv('PUBMED_API_KEY')            # https://account.ncbi.nlm.nih.gov/settings/


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


def pubmed_esearch(query, retmax=100, mindate=None, maxdate=None):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    }
    if mindate: params["mindate"] = mindate
    if maxdate: params["maxdate"] = maxdate
    params["pubmed_api_key"] = pubmed_api_key

    resp = requests.get(pubmed_url + "esearch.fcgi", params=params)
    resp.raise_for_status()
    data = resp.json()
    return data["esearchresult"]["idlist"]


def pubmed_efetch(pmids):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if pubmed_api_key: params["pubmed_api_key"] = pubmed_api_key

    resp = requests.get(pubmed_url + "efetch.fcgi", params=params)
    resp.raise_for_status()
    return resp.text


def parse_pubmed_xml(xml_text):
    root = ET.fromstring(xml_text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None

        art = article.find(".//Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        abstract_els = art.findall(".//AbstractText")

        title = title_el.text if title_el is not None else ""
        abstract = " ".join([a.text for a in abstract_els if a is not None and a.text])

        # Extract the 'pmid' (pubmed article id) also.  Because we can find the url later.
        if abstract.strip():
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
            })

    return articles

 

# ------------------------------------ Load Knowledge Base ----------------------------------
print_banner("Load Knowledge Base")

# Search for article ids that contain the search item and retrieve it as a list
pmids = pubmed_esearch(PUBMED_SEARCH, retmax=50)

# Retrieve the articles as xml
xml_text = pubmed_efetch(pmids)

# Parse the xml into human readable
docs = parse_pubmed_xml(xml_text)

# Explore count
print("Got", len(docs), "articles")

# Save documents to local disc
print("\nSaving documents to local disc")
for doc in docs:
    pmid = doc["pmid"] or "unknown"
    filename = output_dir / f"pmid_{pmid}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        # Extract the title and abstract
        f.write(doc["title"] + "\n\n" + doc["abstract"])
        print(f"✅ Saved to disc: {doc['title']}")



# ------------------------------------ Read in Documents using LangChain's Loader ----------------------------------
print_banner("Read in Documents using LangChain's Loader")

# Read documents from local disc back in with metadata 
print("\nLoading documents back in with metadata")
all_docs = []
for path in output_dir.glob("pmid_*.txt"):
    loader = TextLoader(str(path), encoding="utf-8")
    loaded_docs = loader.load() 

    for d in loaded_docs:
        # Get stem from filename: 'pmid_12345678.txt' -> 'pmid_12345678'
        stem = Path(d.metadata["source"]).stem  # or d.metadata["source"] is full path

        # Get the pmid from the stem: 'pmid_12345678' -> '12345678'
        pmid = stem.split("_", 1)[1] if "_" in stem else None

        # Add meta data to each chunk
        d.metadata["pmid"] = pmid
        d.metadata["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        d.metadata["source_type"] = "pubmed_abstract"

        all_docs.append(d)

        print(f"✅ Loaded in from local disc: {path}")

# Preview the first document
print("\n Example document at index 0:")
print(docs[0])


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

# print(chunks[1])

# Inspect chunks:
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

print(f"*There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store") # There are 123 vectors with 1,536 dimensions in the vector store

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


# # ------------------------------------ 3D Embedding Visualization ----------------------------------
# print_banner("3D Visualization of Embeddings")

# # 1. Compute embeddings for each text chunk
# texts = [c.page_content for c in chunks]
# X = embeddings.embed_documents(texts)
# X = np.array(X)  # shape: (num_chunks, 1536)

# print(f"Embedding matrix: {X.shape}")

# # 2. Reduce dimensions to 3D
# pca = PCA(n_components=3)
# X_3d = pca.fit_transform(X)

# # 3. Plot using matplotlib's 3D scatter
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=40)

# # Label some points with PMID or short identifiers
# for i, doc in enumerate(chunks):
#     pmid = doc.metadata.get("pmid", "")
#     if i % 5 == 0:  # label every 5th to avoid clutter
#         ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], pmid)

# ax.set_title("3D PCA Visualization of PubMed Chunk Embeddings")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")

# plt.tight_layout()
# plt.show()


# --------------------------- 4D Visualization (Color Encoded) ---------------------------
print_banner("4D PCA Visualization (3D Scatter + Color)")

texts = [c.page_content for c in chunks]
X = np.array(embeddings.embed_documents(texts))

# Reduce to 4D
pca = PCA(n_components=4)
X_4d = pca.fit_transform(X)

# First 3 dimensions → axes
x, y, z = X_4d[:, 0], X_4d[:, 1], X_4d[:, 2]
# Fourth dimension → color scale
c = X_4d[:, 3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(x, y, z, c=c, cmap="viridis", s=50)

# Add color bar to show 4th dimension
cbar = plt.colorbar(scatter, label="4th PCA Dimension Value")

ax.set_title("PubMed Embedding Visualization (4D Encoded in Color)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()
