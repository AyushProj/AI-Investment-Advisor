"""Groq chat completions (via Groq API)."""

from __future__ import annotations

import os
from pathlib import Path

# Load `.env` from repo root whenever this module is imported (any page).
try:
    from dotenv import load_dotenv

    _ENV_FILE = Path(__file__).resolve().parent / ".env"
    load_dotenv(_ENV_FILE, override=True)
except ImportError:
    pass

# Default Groq model — fast and capable for investment Q&A
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def get_gemini_model() -> str:
    """Resolved model id — reads GROQ_MODEL env or uses default."""
    v = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    return v if v else DEFAULT_GROQ_MODEL


def get_gemini_api_key() -> str:
    """
    Groq API key from process env, then Streamlit secrets.
    Function name kept as-is so existing callers don't break.
    """
    raw = os.getenv("GROQ_API_KEY", "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    if raw:
        return raw
    try:
        import streamlit as st

        sec = getattr(st, "secrets", None)
        if sec and "GROQ_API_KEY" in sec:
            v = sec["GROQ_API_KEY"]
            return str(v).strip() if v is not None else ""
    except Exception:
        pass
    return ""


def chat_completion(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.3,
) -> str:
    """
    Run a chat completion via Groq.
    Requires GROQ_API_KEY to be set.
    """
    try:
        from groq import Groq
    except ImportError as err:
        raise RuntimeError(
            "Install the Groq SDK: `pip install groq` "
            "(or `pip install -r requirements.txt`)."
        ) from err

    key = get_gemini_api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys"
        )

    client = Groq(api_key=key)

    # Groq supports system messages natively in the messages list
    groq_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        # Groq uses 'assistant' not 'model'
        if role == "model":
            role = "assistant"
        groq_messages.append({"role": role, "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=groq_messages,
        temperature=temperature,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def recommendation_prompt_completion(prompt: str, model: str) -> str:
    return chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
    )