infectious-disease-chat/
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
