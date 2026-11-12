from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .rule_cleanup import clean_transcript_rules

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Ranked for low-latency local prompt cleanup if installed.
PREFERRED_MODELS = (
    "glm-4.7-flash:q4_K_M",
    "gpt-oss:20b",
    "gemma2:27b",
    "qwen3.6:27b",
    "gemma4:31b",
)

CLEANUP_SYSTEM_PROMPT = """You clean up whispered speech-to-text transcripts before they are pasted into an AI coding assistant or chat prompt.

Rules:
1. Fix obvious phonetic errors from whispered speech (e.g. missing/swapped plosives or soft consonants like p/b, t/d, k/g, f/v, s/z) using sentence context.
2. Remove filler words ("um", "uh", "like", "you know") and false starts/stutters, keeping the speaker's exact intent and technical terms intact.
3. Fix punctuation and capitalization cleanly.
4. NEVER invent missing words, NEVER fabricate new sentences from fragments, NEVER answer the prompt, NEVER add explanations, NEVER wrap in quotes or markdown fences.
5. Output ONLY the cleaned transcript text. If the input is empty or unintelligible noise, output an empty string."""


@dataclass(slots=True)
class CleanupResult:
    text: str
    applied: bool
    model: str | None
    error: str | None = None


async def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            payload = resp.json()
            names = [m["name"] for m in payload.get("models", []) if "name" in m]
            return names
    except Exception:
        return []


def pick_default_model(installed: list[str]) -> str | None:
    if not installed:
        return None
    for preferred in PREFERRED_MODELS:
        if preferred in installed:
            return preferred
    return installed[0]


async def clean_transcript(
    raw_text: str,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout_sec: float = 20.0,
) -> CleanupResult:
    stripped = raw_text.strip()
    if not stripped:
        return CleanupResult(text="", applied=False, model=model)

    chosen = model
    if not chosen:
        installed = await list_ollama_models(base_url=base_url)
        chosen = pick_default_model(installed)
    if not chosen:
        return CleanupResult(
            text=stripped,
            applied=False,
            model=None,
            error="No Ollama model available locally.",
        )

    payload = {
        "model": chosen,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 512},
        "messages": [
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": stripped},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            # Some Ollama model builds reject `think: false`; retry cleanly without it if 400.
            if resp.status_code == 400:
                payload.pop("think", None)
                resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            cleaned = (data.get("message", {}).get("content") or "").strip()
            if not cleaned:
                return CleanupResult(text=stripped, applied=False, model=chosen)
            return CleanupResult(text=cleaned, applied=True, model=chosen)
    except Exception as exc:
        return CleanupResult(
            text=stripped,
            applied=False,
            model=chosen,
            error=str(exc),
        )


def _normalize_openai_base_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Base URL is empty.")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Base URL must start with http:// or https://.")
    if parsed.path in ("", "/"):
        return f"{raw}/v1"
    return raw


def _auth_headers(api_key: str | None) -> dict[str, str]:
    token = (api_key or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def list_custom_models(
    base_url: str,
    api_key: str | None = None,
    timeout_sec: float = 6.0,
) -> tuple[list[str], str | None]:
    """Fetch available model IDs from an OpenAI-compatible GET /models endpoint."""
    try:
        norm_base = _normalize_openai_base_url(base_url)
    except ValueError as exc:
        return [], str(exc)

    headers = _auth_headers(api_key)
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.get(f"{norm_base}/models", headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            raw_list = payload.get("data") or payload.get("models") or []
            ids: list[str] = []
            for item in raw_list:
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("name")
                    if isinstance(mid, str) and mid.strip():
                        ids.append(mid.strip())
                elif isinstance(item, str) and item.strip():
                    ids.append(item.strip())
            return sorted(set(ids)), None
    except httpx.HTTPStatusError as exc:
        return [], f"HTTP {exc.response.status_code} from /models"
    except Exception as exc:
        return [], str(exc)


async def clean_transcript_custom_openai(
    raw_text: str,
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    timeout_sec: float = 5.0,
) -> CleanupResult:
    """Rewrite transcript via any OpenAI-compatible /chat/completions endpoint; falls back to rules."""
    stripped = raw_text.strip()
    if not stripped:
        return CleanupResult(text="", applied=False, model=model or None)
    chosen = (model or "").strip()
    if not chosen:
        return CleanupResult(
            text=clean_transcript_rules(stripped),
            applied=True,
            model="rules fallback",
            error="No custom model selected.",
        )
    try:
        norm_base = _normalize_openai_base_url(base_url)
    except ValueError as exc:
        return CleanupResult(
            text=clean_transcript_rules(stripped),
            applied=True,
            model="rules fallback",
            error=str(exc),
        )

    payload = {
        "model": chosen,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": stripped},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(
                f"{norm_base}/chat/completions",
                headers=_auth_headers(api_key),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            content = ""
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = (msg.get("content") or "").strip()
            if not content:
                return CleanupResult(text=clean_transcript_rules(stripped), applied=True, model="rules fallback")
            return CleanupResult(text=content, applied=True, model=chosen)
    except Exception as exc:
        return CleanupResult(
            text=clean_transcript_rules(stripped),
            applied=True,
            model="rules fallback",
            error=str(exc),
        )
