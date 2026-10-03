"""Prompt registry: loads versioned prompts from /prompts and tracks their
content hash (prompt version) for evaluation and audit purposes."""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Final

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

PROMPT_FILES: Final[dict[str, str]] = {
    "router": "router.txt",
    "search": "search.txt",
    "database": "database.txt",
    "general": "general.txt",
    "synthesizer": "synthesizer.txt",
}


@lru_cache
def load_prompt(name: str) -> str:
    """Load a prompt template by agent name. Cached; content is versioned in git."""
    filename = PROMPT_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown prompt: {name}")
    path = _PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache
def prompt_version(name: str) -> str:
    """SHA-256 prefix of the prompt content — the 'prompt version' recorded in agent_runs."""
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()[:12]
