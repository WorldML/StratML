"""
llm_control.py
--------------
Central authoritative controller for LLM invocation across all decision agents.

Rules:
- If llm_mode is explicitly set to False, no LLM call is ever made,
  even if GROQ_API_KEY is present in the environment.
- If llm_mode is explicitly set to True, the LLM is invoked if GROQ_API_KEY is present.
- If llm_mode is None (auto), it defaults to True if GROQ_API_KEY is present, else False.
"""

from __future__ import annotations

import os
from typing import Optional

_LLM_MODE_OVERRIDE: Optional[bool] = None


def set_llm_mode(mode: Optional[bool]) -> None:
    """Explicitly set whether LLM is enabled (True), disabled (False), or auto (None)."""
    global _LLM_MODE_OVERRIDE
    _LLM_MODE_OVERRIDE = mode


def get_llm_mode() -> Optional[bool]:
    """Get the current explicit LLM mode override."""
    return _LLM_MODE_OVERRIDE


def is_llm_enabled() -> bool:
    """Return True if LLM is active and can be invoked."""
    if _LLM_MODE_OVERRIDE is False:
        return False
    if not os.getenv("GROQ_API_KEY"):
        return False
    return True
