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

---
## 📂 03 - Infectious Disease RAG Assistant 🦠

- **Project Overview:**  
  A production-style Retrieval-Augmented Generation (RAG) chat assistant focused on **infectious diseases**, built with a full frontend + backend architecture.  This system combines domain-specific RAG with a **semantic LLM response cache**, **allowing first-turn user questions that are semantically similar to reuse previously generated LLM responses** across users to explore cost savings with messaging LLMs on a large scale.

  This project emphasizes **real-world engineering concerns**: prompt-injection defense, response caching, usage logging, persistence, and UI/backend coordination.
- **Parts:**  
    - Domain-specific RAG - Retrieves infectious-disease context from a FAISS vector store
    - Semantic LLM Response Cache (MySQL + FAISS)
        - Exact-hash cache for repeated first-turn queries  
        - Cache usage is **intentionally limited to first-turn queries** to **avoid applying cached responses in contexts where prior conversation state could change the correct answer**.
        - Semantic cache using cosine similarity over normalized embeddings  
        - Prevents redundant LLM calls for semantically identical questions  
    - Prompt-Injection & Input Hardening
    - Async FastAPI backend orchestrating RAG, caching, and LLM calls  
    - Integration with Amazon Bedrock via async `converse` calls 
    - React-based chat UI
- **Semantic Response Reuse (First-Turn Queries)**
    - The screenshot below illustrates how semantically similar first-turn questions can produce identical responses without requiring a new LLM invocation.  
    - When a user submits an initial question that is sufficiently similar (by cosine similarity over embeddings) to a previously asked question, the system returns a cached response rather than calling the LLM again.  
    - This behavior reduces inference cost while preserving correctness.  
    - The 2nd question is semantically similar enough from the first question to use a recycled LLM response.  However, the 3rd question is just semantically different enough to message the model.  
- **Semantic cache tuning knobs**
    - `CACHE_SIMILARITY_THRESHOLD` (default `0.90`): minimum cosine similarity required to reuse a prior response (higher = safer, lower = more aggressive caching).
    - `CACHE_TOP_K` (default `5`): number of nearest cached queries considered.

<p align="center">
  <img src="https://github.com/david125tran/RAG_System/blob/main/03/UI_01.png?raw=true" width="800" />
</p>

*Example chat demonstrating identical responses for semantically equivalent first-turn questions, enabled by the semantic response cache.*


- **Repo Layout:**
```
03/
│
├── start-frontend.bat                                   # Launch React UI (npm start / vite / etc.)
├── start-backend.bat                                    # Launch FastAPI backend (uvicorn backend.main:app)
│
├── rag pipeline/
│   │
│   ├── knowledge base/
│   │   └── *.txt, *.md, *.pdf, *.docx, *.html, *.htm    # Raw infectious disease reference documents
│   │
│   ├── vectorstore_db/
│   │   ├── index.faiss                                  # FAISS vector index
│   │   └── index.pkl                                    # Serialized metadata
│   │
│   └── vectorstore_builder.py                           # Script to embed documents & build FAISS DB
│
├── backend/
│   │
│   ├── main.py                                          # FastAPI entry point (POST /api/chat)
│   ├── aws_bedrock_client.py                            # Async AWS Bedrock chat wrapper
│   ├── validate_input.py                                # Prompt-injection & safety validation
│   │
│   ├── .env                                             # Environment variables:
│   │                                                     #  - AWS_ACCESS_KEY_ID
│   │                                                     #  - AWS_SECRET_ACCESS_KEY
│   │                                                     #  - OPENAI_API_KEY
│   │                                                     #  - BASE_MODEL
│   │                                                     #  - DB_USER / DB_PW / DB_DATABASE_NAME
│   │
│   └── __pycache__/                                     # Python bytecode cache
│
├── frontend/
│   │
│   ├── package.json                                     # Frontend dependencies & scripts
│   ├── node_modules/
│   │
│   ├── public/
│   │   ├── index.html                                   # HTML shell
│   │   └── avatar.png                                   # Assistant avatar
│   │
│   └── src/
│       ├── App.js                                       # Main chat UI (state, API calls, history)
│       ├── App.css                                      # Chat UI styling
│       ├── index.js                                     # React entry point → App
│       └── index.css                                    # Global styles (resets, fonts)
│
└── directory.md                                         # Project directory overview (this file)
```
