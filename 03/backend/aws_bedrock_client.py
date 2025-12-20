# 03/backend/aws_bedrock_client.py

import asyncio
import boto3
from dataclasses import dataclass
import os
import re
from typing import Optional, List, Dict, Any



# ---------------------------------- Output Santization ----------------------------------
SECRET_PATTERNS = [
    # OpenAI-style Keys
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    # Misc. Keys
    re.compile(r"(?i)aws_?(secret|access)_?key\s*[:=]\s*[A-Za-z0-9/+=]{20,}"),
    # AWS Access Key IDs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b")
]

def sanitize_model_output(text: str) -> str:
    """
    Remove / mask anything we never want to leak to the user.
    """
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED_SECRET]", text)

    return text



# ---------------------------------- AWS Bedrock Client ----------------------------------
@dataclass
class BedrockChatResult:
    reply: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    model_id: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class BedrockClient:
  """
  Thin async wrapper around the Amazon Bedrock Runtime `converse` API
  for different LLM models.
  """

  def __init__(
    self,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
  ) -> None:
    session_kwargs: Dict[str, Any] = {}
    if profile_name:
      session_kwargs["profile_name"] = profile_name

    # Create the boto3 session AFTER env vars are loaded
    session = boto3.Session(**session_kwargs) if session_kwargs else boto3.Session()

    # Debug what credentials boto3 is using
    creds = session.get_credentials()
    if creds:
      frozen = creds.get_frozen_credentials()
      print("=== AWS DEBUG ===")
      print("Access key starts with:", frozen.access_key[:4], "****")
      print("Cred provider:", creds.method)
      print("=================")
    else:
      print("=== AWS DEBUG ===")
      print("No credentials resolved!")
      print("=================")

    # Region: from argument, then env var, then default
    resolved_region = region_name or os.getenv("AWS_REGION") or "us-east-1"

    self.client = session.client(
      "bedrock-runtime",
      region_name=resolved_region,
    )

  # ---------- Internal Sync Helper ----------
  def _converse_sync(
    self,
    *,
    model: str,
    system: Optional[str],
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    tools: Optional[List[Dict[str, Any]]] = None,
  ) -> BedrockChatResult:
    """
    Synchronous call to Bedrock Converse, returns assistant text.
    """

    messages: List[Dict[str, Any]] = []

    # Add history, but skip blank content
    if history:
      for turn in history:
        # Support either {role, content} or {from, text}
        if "content" in turn:
          raw_content = turn.get("content")
        else:
          raw_content = turn.get("text")

        content = (raw_content or "").strip()
        if not content:
          # Skip empty/whitespace-only history entries
          continue

        if "role" in turn:
          role = turn.get("role", "user")
        else:
          from_ = turn.get("from", "user")
          role = "user" if from_ == "user" else "assistant"

        messages.append(
          {
            "role": role,
            "content": [{"text": content}],
          }
        )

    # Add current user message (must not be blank)
    user_text = (message or "").strip()
    if not user_text:
      raise ValueError("Current user message cannot be empty or whitespace")

    messages.append(
      {
        "role": "user",
        "content": [{"text": user_text}],
      }
    )

    kwargs: Dict[str, Any] = {
      "modelId": model,
      "messages": messages,
      "inferenceConfig": {
        "maxTokens": max_tokens,
        "temperature": temperature,
      },
    }

    # Clean system prompt too
    system_text = (system or "").strip()
    if system_text:
      kwargs["system"] = [{"text": system_text}]

    if tools:
      kwargs["toolConfig"] = {"tools": tools}

    response = self.client.converse(**kwargs)

    output_msg = response.get("output", {}).get("message", {})
    text_chunks: List[str] = []
    for item in output_msg.get("content", []):
      if "text" in item:
        text_chunks.append(item["text"])

    raw = "".join(text_chunks).strip()
    safe = sanitize_model_output(raw)

    usage = response.get("usage", {}) or {}
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    total_tokens = usage.get("totalTokens")

    stop_reason = response.get("stopReason") or response.get("stop_reason")
    model_id = response.get("modelId") or kwargs.get("modelId")

    return BedrockChatResult(
      reply=safe,
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      total_tokens=total_tokens,
      stop_reason=stop_reason,
      model_id=model_id,
      raw_response=None,  # set to response if you want
    )

  # ---------- Public async helper used by pipelines ----------
  async def chat(
    self,
    *,
    model: str,
    system: Optional[str],
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    tools: Optional[List[Dict[str, Any]]] = None,
  ) -> BedrockChatResult:
    """
    Async wrapper around _converse_sync so you can `await` it
    from FastAPI / any async code.
    """
    return await asyncio.to_thread(
      self._converse_sync,
      model=model,
      system=system,
      message=message,
      history=history,
      max_tokens=max_tokens,
      temperature=temperature,
      tools=tools,
    )


# Default instance used by the rest of the app
def make_bedrock_client() -> BedrockClient:
    return BedrockClient()