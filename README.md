# 📚 Retrieval-Augmented Generation (RAG) Systems 🧪
## About
* This repo is a sandbox for different RAG System projects.  
* Each folder `01`, `02`, etc. is a different project.  

---
## 01 - SEC 10‑K filings 🏛️
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




