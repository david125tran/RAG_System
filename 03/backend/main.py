# 03/backend/main.py

# ------------------------------------ Libraries ----------------------------------
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import mysql.connector
import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple
import uuid
import hashlib
import threading
import time

import numpy as np
import faiss

from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_openai import OpenAIEmbeddings

try:
    from .validate_input import validate_query  # when run as package (uvicorn backend.main:app)
except ImportError:
    from validate_input import validate_query   # when run as script (python main.py)

try:
    from .aws_bedrock_client import make_bedrock_client
except ImportError:
    from aws_bedrock_client import make_bedrock_client


# ------------------------------------ Variables / Constants ----------------------------------
# Script Directory
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

# Vector DB Path
vector_store = parent_dir / "rag pipeline" / "vectorstore_db"

# Semantic Cache Tuning
CACHE_TOP_K = 5
CACHE_SIMILARITY_THRESHOLD = 0.90  # cosine similarity (0.1), higher = more similar
CACHE_MAX_ENTRIES_LOAD = 5000      # safety cap on startup load
CACHE_REBUILD_COOLDOWN_SEC = 15    # throttle rebuilds if needed


# ------------------------------------ Configure API Keys / Tokens ----------------------------------
env_path = parent_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

aws_bedrock_client = make_bedrock_client()

openai_api_key = os.getenv("OPENAI_API_KEY")
base_model = os.getenv("BASE_MODEL")


# ------------------------------------ DB Config ----------------------------------
DB_USER = os.getenv("DB_USER")
DB_PW = os.getenv("DB_PW")
DB_DATABASE_NAME = os.getenv("DB_DATABASE_NAME")
DB_CACHE_TABLE_NAME = os.getenv("DB_CACHE_TABLE_NAME")
DB_USAGE_TABLE_NAME = os.getenv("DB_USAGE_TABLE_NAME")

if not all([openai_api_key, base_model, DB_USER, DB_PW, DB_DATABASE_NAME, DB_CACHE_TABLE_NAME, DB_USAGE_TABLE_NAME]):
    raise RuntimeError("Missing one or more required env vars. Check your .env.")


# ------------------------------------ DB Functions ----------------------------------
def get_db_connection():
    """
    Connect to MySQL WITH the target database selected.
    """
    return mysql.connector.connect(
        user=DB_USER,
        password=DB_PW,
        database=DB_DATABASE_NAME,
        autocommit=True,
    )


def create_database_if_missing():
    """
    Create the MySQL db if it is non-existent
    """
    db = DB_DATABASE_NAME
    ddl = f"""
    CREATE DATABASE IF NOT EXISTS `{db}`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
    """

    cnx = mysql.connector.connect(
        user=DB_USER,
        password=DB_PW,
        autocommit=True,
    )

    try:
        with cnx.cursor() as cur:
            cur.execute(ddl)
        print(f"Database ensured: {db}")
    finally:
        cnx.close()


def column_exists(table: str, column: str) -> bool:
    """
    Check if the column exists and return True or False
    """
    sql = """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
    """
    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(sql, (DB_DATABASE_NAME, table, column))
            (cnt,) = cur.fetchone()
            return int(cnt) > 0
    finally:
        cnx.close()


def ensure_column(table: str, column_ddl: str, column_name: str):
    """
    Add a column if it is missing.
    """
    # Prune
    if column_exists(table, column_name):
        return
    
    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(f"ALTER TABLE `{table}` ADD COLUMN {column_ddl};")
        print(f"Added column {column_name} to {table}")
    finally:
        cnx.close()


def create_cache_table_if_missing():
    """
    Cache table:
      - created_at:     timestamp of row entry
      - user_query:     user's question
      - llm_response:   llm's response
      - query_hash:     fast lookup/dedup
      - model:          which model produced the response
      - embedding_json: JSON-encoded embedding vector for semantic search
    """
    table = DB_CACHE_TABLE_NAME

    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        user_query TEXT NOT NULL,
        llm_response MEDIUMTEXT NOT NULL,
        query_hash BINARY(32) NULL,
        model VARCHAR(128) NULL,
        embedding_json MEDIUMTEXT NULL,

        PRIMARY KEY (id),
        INDEX idx_created_at (created_at),
        INDEX idx_query_hash (query_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(ddl)
        print("Cache table ensured.")
    finally:
        cnx.close()

    # lightweight migrations in case table existed before this change
    ensure_column(table, "embedding_json MEDIUMTEXT NULL", "embedding_json")


def create_usage_table_if_missing():
    """
    Usage table:
      - id:                     row id
      - created_at:             timestamp of row entry
      - flagged                 if query was flagged
      - user_query              user's question
      - input_tokens            input tokens
      - output_tokens           output tokens
      - model                   model used (base_model)
      - was_cache_hit           if question:response pair was cached
      - error                   store exception text if the call fails
      - user_ip                 ip address of user
      - llm_response            llm's response (cached or model-generated)
    """
    table = DB_USAGE_TABLE_NAME

    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        flagged TINYINT(1) NOT NULL DEFAULT 0,
        user_query TEXT NOT NULL,
        input_tokens INT UNSIGNED NULL,
        output_tokens INT UNSIGNED NULL,
        model VARCHAR(128) NULL,
        was_cache_hit TINYINT(1) NOT NULL DEFAULT 0,
        error TEXT NULL,
        user_ip VARCHAR(45) NULL,
        llm_response MEDIUMTEXT NOT NULL,

        PRIMARY KEY (id),
        INDEX idx_created_at (created_at),
        INDEX idx_flagged_created_at (flagged, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(ddl)
        print("Usage table ensured.")
    finally:
        cnx.close()


def log_usage_question(
    user_query: str,
    *,
    flagged: bool = False,
    model: Optional[str] = None,
    user_ip: Optional[str] = None,
    was_cache_hit: bool = False,
    request_id: Optional[str] = None,
) -> int:
    """
    Insert a usage row as soon as we receive the user's question.
    """

    table = DB_USAGE_TABLE_NAME
    request_id = request_id or str(uuid.uuid4())
    model = model or base_model

    sql = f"""
        INSERT INTO `{table}`
        (created_at, flagged, user_query, input_tokens, output_tokens, model, was_cache_hit, error, user_ip, llm_response)
        VALUES (CURRENT_TIMESTAMP, %s, %s, NULL, NULL, %s, %s, NULL, %s, %s)
    """

    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(sql, (
                1 if flagged else 0,
                user_query,
                model,
                1 if was_cache_hit else 0,
                user_ip,
                "",  # placeholder response
            ))
            return cur.lastrowid
    finally:
        cnx.close()


def update_usage_row(
    row_id: int,
    *,
    llm_response: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[str] = None,
    flagged: Optional[bool] = None,
    was_cache_hit: Optional[bool] = None,
) -> None:
    """
    Update an existing usage row with response/tokens/error/cache_hit.  It does a partial
    update on a single row in the usage table.  Only the fields passed in are updated.  
    Fields that are lefts as `None` are untouched.  
    
    """
    table = DB_USAGE_TABLE_NAME

    # The columns to update
    updates = []
    # The data that is passed into the row's cell
    params = []

    # Update clauses dynamically
    if llm_response is not None:
        updates.append("llm_response = %s")
        params.append(llm_response)

    if input_tokens is not None:
        updates.append("input_tokens = %s")
        params.append(int(input_tokens))

    if output_tokens is not None:
        updates.append("output_tokens = %s")
        params.append(int(output_tokens))

    if error is not None:
        updates.append("error = %s")
        params.append(error)

    if flagged is not None:
        updates.append("flagged = %s")
        params.append(1 if flagged else 0)

    if was_cache_hit is not None:
        updates.append("was_cache_hit = %s")
        params.append(1 if was_cache_hit else 0)

    if not updates:
        return

    sql = f"UPDATE `{table}` SET " + ", ".join(updates) + " WHERE id = %s"
    params.append(int(row_id))

    cnx = get_db_connection()

    try:
        with cnx.cursor() as cur:
            cur.execute(sql, tuple(params))
    finally:
        cnx.close()


# ------------------------------------ Semantic Cache Helpers ----------------------------------
def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = " ".join(q.split())
    return q


def sha256_bin32(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(api_key=openai_api_key)


def embed_query(text: str) -> np.ndarray:
    """Returns L2-normalized float32 embedding vector."""
    emb = get_embeddings().embed_query(text)
    v = np.array(emb, dtype=np.float32)
    # L2 normalize so inner product == cosine similarity
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def fetch_cache_by_hash(query_hash: bytes) -> Optional[Tuple[int, str, str, Optional[str], Optional[str]]]:
    """Exact hash match (fast path)."""
    table = DB_CACHE_TABLE_NAME
    sql = f"""
        SELECT id, user_query, llm_response, model, embedding_json
        FROM `{table}`
        WHERE query_hash = %s
        ORDER BY id DESC
        LIMIT 1
    """
    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(sql, (query_hash,))
            row = cur.fetchone()
            if not row:
                return None
            return int(row[0]), row[1], row[2], row[3], row[4]
    finally:
        cnx.close()


def insert_cache_entry(
    user_query: str,
    llm_response: str,
    *,
    model: Optional[str] = None,
    embedding_json: Optional[str] = None,
    query_hash: Optional[bytes] = None,
) -> int:
    table = DB_CACHE_TABLE_NAME
    model = model or base_model

    sql = f"""
        INSERT INTO `{table}` (created_at, user_query, llm_response, query_hash, model, embedding_json)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)
    """

    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(sql, (
                user_query,
                llm_response,
                query_hash,
                model,
                embedding_json,
            ))
            return cur.lastrowid
    finally:
        cnx.close()


def fetch_cache_entries_for_index(limit: int) -> List[Tuple[int, str, str, Optional[str]]]:
    """
    Fetch cached rows to build in-memory semantic index.
    """
    table = DB_CACHE_TABLE_NAME
    sql = f"""
        SELECT id, user_query, llm_response, embedding_json
        FROM `{table}`
        ORDER BY id DESC
        LIMIT %s
    """
    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(sql, (int(limit),))
            rows = cur.fetchall() or []
            return [(int(r[0]), r[1], r[2], r[3]) for r in rows]
    finally:
        cnx.close()


def backfill_embedding_if_missing(row_id: int, embedding_json: str) -> None:
    table = DB_CACHE_TABLE_NAME
    sql = f"UPDATE `{table}` SET embedding_json = %s WHERE id = %s AND embedding_json IS NULL"
    cnx = get_db_connection()
    try:
        with cnx.cursor() as cur:
            cur.execute(sql, (embedding_json, int(row_id)))
    finally:
        cnx.close()


# In-memory FAISS index for semantic cache
_cache_lock = threading.Lock()
_cache_index: Optional[faiss.IndexFlatIP] = None
_cache_meta: List[Dict[str, Any]] = []
_cache_dim: Optional[int] = None
_cache_last_rebuild: float = 0.0


def rebuild_cache_index_if_needed(force: bool = False) -> None:
    """
    Build/refresh in-memory FAISS index from MySQL cache table.
    Uses cosine similarity via IndexFlatIP with L2-normalized vectors.
    """
    global _cache_index, _cache_meta, _cache_dim, _cache_last_rebuild

    now = time.time()
    if not force and (now - _cache_last_rebuild) < CACHE_REBUILD_COOLDOWN_SEC and _cache_index is not None:
        return

    with _cache_lock:
        now = time.time()
        if not force and (now - _cache_last_rebuild) < CACHE_REBUILD_COOLDOWN_SEC and _cache_index is not None:
            return

        rows = fetch_cache_entries_for_index(CACHE_MAX_ENTRIES_LOAD)

        vectors: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []

        for row_id, q, resp, emb_json in rows:
            q_norm = normalize_query(q)
            if not emb_json:
                # backfill embedding once (costs embedding call) so future loads are cheap
                v = embed_query(q_norm)
                emb_json = json.dumps(v.tolist())
                backfill_embedding_if_missing(row_id, emb_json)
            else:
                try:
                    v_list = json.loads(emb_json)
                    v = np.array(v_list, dtype=np.float32)
                    # ensure normalized
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        v = v / norm
                except Exception:
                    v = embed_query(q_norm)
                    emb_json = json.dumps(v.tolist())
                    backfill_embedding_if_missing(row_id, emb_json)

            vectors.append(v)
            meta.append({"id": row_id, "user_query": q, "llm_response": resp})

        if not vectors:
            _cache_index = None
            _cache_meta = []
            _cache_dim = None
            _cache_last_rebuild = time.time()
            return

        dim = int(vectors[0].shape[0])
        mat = np.vstack(vectors).astype(np.float32)

        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        _cache_index = index
        _cache_meta = meta
        _cache_dim = dim
        _cache_last_rebuild = time.time()


def cache_semantic_lookup(user_query: str) -> Optional[Tuple[str, float, int]]:
    """
    Returns (cached_response, similarity, cache_row_id) if semantic match >= threshold.
    Similarity is cosine similarity in [0, 1] (practically can be negative if unrelated).
    """
    q_norm = normalize_query(user_query)

    rebuild_cache_index_if_needed(force=False)

    with _cache_lock:
        if _cache_index is None or _cache_dim is None or not _cache_meta:
            return None

        qv = embed_query(q_norm).reshape(1, -1)
        if qv.shape[1] != _cache_dim:
            # embedding model changed; force rebuild
            return None

        k = min(CACHE_TOP_K, len(_cache_meta))
        scores, idxs = _cache_index.search(qv.astype(np.float32), k)

        best_score = float(scores[0][0]) if k > 0 else -1.0
        best_idx = int(idxs[0][0]) if k > 0 else -1

        if best_idx < 0 or best_score < CACHE_SIMILARITY_THRESHOLD:
            return None

        hit = _cache_meta[best_idx]
        return hit["llm_response"], best_score, int(hit["id"])


def cache_add_entry_incremental(cache_id: int, user_query: str, llm_response: str, embedding_vec: np.ndarray) -> None:
    """Incrementally add a new cached item into the in-memory FAISS index."""
    global _cache_index, _cache_meta, _cache_dim

    with _cache_lock:
        v = embedding_vec.astype(np.float32)
        if _cache_index is None:
            _cache_dim = int(v.shape[0])
            _cache_index = faiss.IndexFlatIP(_cache_dim)
            _cache_index.add(v.reshape(1, -1))
            _cache_meta = [{"id": cache_id, "user_query": user_query, "llm_response": llm_response}]
            return

        if _cache_dim != int(v.shape[0]):
            # embedding model/dim changed; safest is rebuild later
            return

        _cache_index.add(v.reshape(1, -1))
        _cache_meta.append({"id": cache_id, "user_query": user_query, "llm_response": llm_response})


# ------------------------------------ Functions ----------------------------------
def convert_history_for_bedrock(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert React-style history:
        [{ "from": "user" | "bot", "text": "..." }, ...]
    or Bedrock-style:
        [{ "role": "user" | "assistant", "content": "..." }, ...]
    into the format expected by BedrockClient._converse_sync:
        [{ "role": "user" | "assistant", "content": "..." }, ...]
    """
    if not history:
        return []

    converted: List[Dict[str, Any]] = []
    for turn in history:
        if "role" in turn or "content" in turn:
            raw_text = turn.get("content", "")
            role = turn.get("role", "user")
        else:
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
    return LC_FAISS.load_local(str(vector_store), embeddings, allow_dangerous_deserialization=True)


# ------------------------------------ Chat Classes ----------------------------------
class ChatRequest(BaseModel):
    backendId: str
    message: str
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    reply: str


# ------------------------------------ Server Side Database Setup ----------------------------------
create_database_if_missing()
create_cache_table_if_missing()
create_usage_table_if_missing()

# Build semantic cache index once at startup (cheap if embeddings already stored)
try:
    rebuild_cache_index_if_needed(force=True)
except Exception as e:
    print(f"[WARN] Cache index build failed at startup: {e}")


# ------------------------------------ Server Side Python Backend ----------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    message: str = Form(""),
    history: str = Form("[]"),
):
    """
    Chat endpoint that supports text + history.
    The frontend sends multipart/form-data (FormData).
    """

    # Extract client's ip
    client_ip = request.client.host if request.client else None

    # ----------------- Prompt-injection / security validation -----------------
    ok, sanitized_message, error_msg = validate_query(message)

    # Cases where the message was flagged by `validate_query()`
    if not ok:
        # Log the flagged message
        log_usage_question(
            user_query=message,
            flagged=True,
            model=base_model,
            user_ip=client_ip,
            was_cache_hit=False,
        )
        return ChatResponse(reply=error_msg)

    if sanitized_message is not None:
        message = sanitized_message
    
    
    # ----------------- Parse history JSON -----------------
    try:
        history_list = json.loads(history) if history else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history JSON")

    # See if this is the user's first message
    is_first_turn = not history_list  # True if [] or None

    # Log question immediately
    usage_row_id = log_usage_question(
        user_query=message,
        flagged=False,
        model=base_model,
        user_ip=client_ip,
        was_cache_hit=False,
    )

    # ----------------- Semantic Cache Lookup (Before Calling LLM) -----------------
    # Only use recycled LLM responses from the cache if it is the user's first message to the chat.
    # This removes edge cases where the user may message the LLM questions using historical context
    # and introduce edge cases where this caching may not work.

    if is_first_turn:
        try:
            q_norm = normalize_query(message)
            q_hash = sha256_bin32(q_norm)

            # Fast exact match
            exact = fetch_cache_by_hash(q_hash)
            if exact:
                _, _, cached_resp, _, _ = exact
                update_usage_row(
                    usage_row_id,
                    llm_response=cached_resp,
                    was_cache_hit=True,
                    input_tokens=0,
                    output_tokens=0,
                )
                return ChatResponse(reply=cached_resp)

            # Semantic match
            sem_hit = cache_semantic_lookup(message)
            if sem_hit:
                cached_resp, sim, cache_id = sem_hit
                update_usage_row(
                    usage_row_id,
                    llm_response=cached_resp,
                    was_cache_hit=True,
                    input_tokens=0,
                    output_tokens=0,
                )
                return ChatResponse(reply=cached_resp)

        except Exception as e:
            # Cache should never take down the endpoint; fall back to LLM.
            update_usage_row(usage_row_id, error=f"[cache_warning] {e}")

    # ---------------------------------------------------------------------------

    # ----------------- RAG retrieval + LLM call (MISS) -----------------
    system_prompt = """
    You are an expert assistant specializing in infectious diseases.

    You answer user questions using the provided context when it is relevant.
    Base your responses primarily on the supplied context, and use general domain
    knowledge only to clarify or connect information already present.

    If the context does not contain enough information to answer the question,
    state that clearly rather than guessing or inventing details.

    Be concise, accurate, and scientifically grounded.
    """

    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(message, k=4)
    context = "\n\n".join(doc.page_content for doc in results)

    user_message = f"""Question:\n{message}\n\nContext:\n<<<\n{context}\n>>>"""

    bedrock_history = convert_history_for_bedrock(history_list)

    try:
        result = await aws_bedrock_client.chat(
            model=base_model,
            system=system_prompt,
            message=user_message,
            history=bedrock_history,
        )

        # Update usage
        update_usage_row(
            usage_row_id,
            llm_response=result.reply,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            was_cache_hit=False,
        )

        # Insert into cache only if it wasn't already in the bank and if the user is still
        # on the first message to the LLM
        if is_first_turn:
            try:
                q_norm = normalize_query(message)
                q_hash = sha256_bin32(q_norm)
                q_vec = embed_query(q_norm)
                emb_json = json.dumps(q_vec.tolist())

                cache_id = insert_cache_entry(
                    user_query=message,
                    llm_response=result.reply,
                    model=base_model,
                    embedding_json=emb_json,
                    query_hash=q_hash,
                )

                # Keep in-memory cache fresh
                cache_add_entry_incremental(cache_id, message, result.reply, q_vec)

            except Exception as e:
                # Don't fail the request if cache write fails
                update_usage_row(usage_row_id, error=f"[cache_write_warning] {e}")

        # Return the LLM response
        return ChatResponse(reply=result.reply)

    except Exception as e:
        update_usage_row(usage_row_id, error=str(e))
        raise
