"""LLM access through free-tier providers.

Provider options (set LLM_PROVIDER in .env):
  - groq         : free tier, extremely fast Llama 3.1 (recommended). Key: console.groq.com
  - huggingface  : free tier via HF Inference router. Token: huggingface.co/settings/tokens

Both expose an OpenAI-compatible /chat/completions endpoint, so one client covers both.
"""
import json
from typing import AsyncGenerator

import httpx

from backend.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GROQ_URL,
    HF_TOKEN, HF_MODEL, HF_URL,
)


def _endpoint() -> tuple[str, str, str]:
    if LLM_PROVIDER == "huggingface":
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is missing. Add it to your .env file (free at huggingface.co/settings/tokens).")
        return HF_URL, HF_TOKEN, HF_MODEL
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file (free at console.groq.com).")
    return GROQ_URL, GROQ_API_KEY, GROQ_MODEL


def _payload(messages: list[dict], stream: bool, max_tokens: int = 900) -> dict:
    _, _, model = _endpoint()
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": stream,
    }


async def complete(messages: list[dict], max_tokens: int = 900) -> str:
    url, key, _ = _endpoint()
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=_payload(messages, stream=False, max_tokens=max_tokens),
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def stream(messages: list[dict], max_tokens: int = 900) -> AsyncGenerator[str, None]:
    """Yield answer tokens as they arrive (SSE from provider -> generator)."""
    url, key, _ = _endpoint()
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=_payload(messages, stream=True, max_tokens=max_tokens),
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
