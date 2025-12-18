# 03/backend/validate_input.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from typing import Any, Iterable, List, Optional


# ------------------------------------ Variables / Constants ----------------------------------
DANGEROUS_WORDS = re.compile(
    r"(?i)\b(ignore previous|override.*instructions|system prompt|jailbreak|"
    r"developer mode|bypass|prompt injection|sudo rm -rf|;--|/\*|\*/|xp_cmdshell)\b"
)
SQL_SIGNS = re.compile(r"(?i)\b(UNION SELECT|--|#|/\*|\*/|;|DROP|INSERT|UPDATE|DELETE)\b")

URL_RE = re.compile(r"https?://[^\s)>\]]+")

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)aws_?(secret|access)_?key\s*[:=]\s*[A-Za-z0-9/+=]{20,}"),
]


# ---------------------------------- Prompt Injection Attack Defense Functions ----------------------------------
def validate_query(query: str):
    """
    Returns:
      (ok: bool, sanitized_query: Optional[str], error_msg: Optional[str])
    """
    # Hard-block obvious prompt / SQL injection
    for pat in (DANGEROUS_WORDS, SQL_SIGNS):
        if pat.search(query):
            return (
                False,
                None,
                "⚠️ Your previous message was blocked because it looked like "
                "a prompt / SQL injection attempt. Please rephrase and try again."
            )

    # Redact any secrets if they appear
    sanitized = query
    for pat in SECRET_PATTERNS:
        sanitized = pat.sub("[REDACTED_SECRET]", sanitized)

    return True, sanitized, None

    
