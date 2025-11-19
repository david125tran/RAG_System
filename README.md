# 📚 Retrieval-Augmented Generation (RAG) Systems 🧪
## About
* This repo is a sandbox for different RAG System projects.  
* Each folder `01`, `02`, etc. is a different project.  

---
## 📂 01 - SEC 10‑K filings RAG System 🏛️
- **Project Overview:** An end-to-end Retrieval-Augmented Generation (RAG) system for querying SEC 10‑K filings. 
- **Dataset:** 🔗 [S&P 500 EDGAR 10-K (Hugging Face)](https://huggingface.co/datasets/jlohding/sp500-edgar-10k)
    - Original filings from SEC EDGAR; mirrored on Hugging Face for research use.
- **Parts:** 
    - **`Create_VectorDB.ipynb`** - Google Colab: Vector Store Builder - chunk + embed 10-Ks and write a persistent **ChromaDB** collection (`edgar_10k`) using `BAAI/bge-small-en-v1.5`. Built on a GPU runtime (A100) for speed.
    - **`edgar_10k_chroma_*`** - Download Vector Store - copy the Chroma folder from Colab to your local machine.
    - **`RAG Pipeline.py`** - Local Machine: RAG Pipeline - retrieve top chunks and answer with a small local LLM (`Qwen/Qwen2.5-0.5B-Instruct`) on CPU (my local machine that doesn't have GPU runtime).
- **Repo Layout:**
```
01/
├── Create_VectorDB.ipynb
├── RAG Pipeline.py
└── VectorDB/
   └── edgar_10k_chroma_*/
      └── edgar_10k_chroma_YYYYMMDD_HHMMSS/               
      └── chroma.sqlite3.sqlite3
```

- **Limitations (by design):** This project has some limitations that I'm aware of.  However, I was using this to play around with RAG concepts and learn more about different frameworks.  
    - Company parsing is heuristic (regex); sensitive to spelling, punctuation, and renames.
    - Tiny LLM (Qwen 0.5B) → basic reasoning only; answers strictly from context.
---

## 📂 02 - Biomedical PubMed RAG Assistant 🧬  
- **Project Overview:**  
  A Retrieval-Augmented Generation (RAG) system that automatically retrieves biomedical research from PubMed based on a topic (e.g., *plasma metanephrines*), downloads abstracts, chunks and embeds them, builds a **FAISS** vector store, and answers domain-specific biomedical queries grounded in real literature.

- **Data Source:** 🔗 [PubMed E-Utilities API ](https://www.ncbi.nlm.nih.gov/home/develop/api/)
  - Articles retrieved programmatically via `esearch` + `efetch`.  
  - Abstracts parsed from XML and stored locally with metadata (PMID, source URL).

- **Parts:**  
  - **`RAG_PubMed_Pipeline.py`** - End-to-end script that:
    - Queries PubMed API  
    - Downloads/Stores abstracts  
    - Splits content → embeddings → **FAISS VectorDB**  
    - Runs a RAG pipeline using `ChatOpenAI` + LangChain retrieval  
    - Optionally visualizes embeddings using **PCA** in 3D or 4D
  - **`pubmed_abstracts/`** - Folder containing normalized article text files organized by search topic.
  - **`vectorstore_db/`** - Local FAISS vector store persisted to disk.

- **Repo Layout:**
```
02/
├── RAG_PubMed_Pipeline.py
├── pubmed_abstracts/
│ └── plasma metanephrine/
│ └── pmid_*.txt
└── vectorstore_db/
└── index.faiss + metadata
```
- **Limitations (by design):** I could only get access to the abstracts of research papers in PubMed.  Full-text access to articles varies by journal licensing, so some responses may lack deeper context available only in complete articles.  I only asked it questions that I knew came from the abstracts.  
