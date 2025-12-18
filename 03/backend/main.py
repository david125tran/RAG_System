# 03/backend/main.py
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from pydantic import BaseModel
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from .validate_input import validate_query  # when run as package (uvicorn backend.main:app)
except ImportError:
    from validate_input import validate_query   # when run as script (python main.py)

try:
    from .aws_bedrock_client import make_bedrock_client
except ImportError:
    from aws_bedrock_client import make_bedrock_client

# ------------------------------------ Variables / Constants ----------------------------------
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

vector_store = parent_dir / "rag pipeline" / "vectorstore_db"


# ------------------------------------ Configure API Keys / Tokens ----------------------------------
# Path to the .env file
env_path = parent_dir / ".env"

# Load the .env file
load_dotenv(dotenv_path=env_path, override=True)

aws_bedrock_client = make_bedrock_client()

# Access the API keys stored in the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")  # https://openai.com/api/

base_model = os.getenv("BASE_MODEL")

print(openai_api_key[:10])
print(base_model[:10])


# ------------------------------------ Functions ----------------------------------
def convert_history_for_bedrock(
    history: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Convert React-style history:
        [{ "from": "user" | "bot", "text": "..." }, ...]
    or Bedrock-style:
        [{ "role": "user" | "assistant", "content": "..." }, ...]
    into the format expected by BedrockClient._converse_sync:
        [{ "role": "user" | "assistant", "content": "..." }, ...]
    (the client itself wraps content as {"text": ...}).
    """
    if not history:
        return []

    converted: List[Dict[str, Any]] = []

    for turn in history:
        # Prefer explicit role/content if already present
        if "role" in turn or "content" in turn:
            raw_text = turn.get("content", "")
            role = turn.get("role", "user")
        else:
            # React-style: { from, text }
            from_ = turn.get("from", "user")
            raw_text = turn.get("text", "")
            role = "user" if from_ == "user" else "assistant"

        text = (raw_text or "").strip()
        if not text:
            continue

        converted.append({"role": role, "content": text})

    return converted


@lru_cache(maxsize=1)
def get_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    return FAISS.load_local(str(vector_store), embeddings, allow_dangerous_deserialization=True)


# ------------------------------------ Chat Classes ----------------------------------
class ChatRequest(BaseModel):
    backendId: str
    message: str
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str


# ------------------------------------ Server Side Python Backend ----------------------------------
app = FastAPI()

# CORS so React can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    message: str = Form(""),
    history: str = Form("[]"),
):
    """
    Chat endpoint that supports text, history, and an optional uploaded file.
    The frontend sends multipart/form-data (FormData).
    """

    # ----------------- Prompt-injection / security validation -----------------
    ok, sanitized_message, error_msg = validate_query(message)
    if not ok:
        # Respond like a normal assistant reply so the user sees it in chat
        return ChatResponse(reply=error_msg)

    if sanitized_message is not None:
        message = sanitized_message
    # -------------------------------------------------------------------------

    # Parse history JSON
    try:
        history_list = json.loads(history) if history else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history JSON")


    system_prompt = f"""
    You are an expert assistant specializing in infectious diseases.

    You answer user questions using the provided context when it is relevant.
    Base your responses primarily on the supplied context, and use general domain
    knowledge only to clarify or connect information already present.

    If the context does not contain enough information to answer the question,
    state that clearly rather than guessing or inventing details.

    Be concise, accurate, and scientifically grounded.
    """

    # Load FAISS vector store
    vectorstore = get_vectorstore()

    # Retrieve relevant documents using similarity search
    results = vectorstore.similarity_search(message, k=4)

    # Extract & join the context from documents
    context = "\n\n".join(doc.page_content for doc in results)


    # Build a plain user message string including context + uploaded file excerpt
    user_message = f"""Question:
    {message}

    Context:
    <<<
    {context}
    >>>
    """

    # Convert history for Bedrock
    bedrock_history = convert_history_for_bedrock(history_list)

    # Call model
    completion = await aws_bedrock_client.chat(
        model=base_model,
        system=system_prompt,
        message=user_message,
        history=bedrock_history,
    )

    return ChatResponse(reply=completion)