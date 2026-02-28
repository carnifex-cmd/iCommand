"""Configuration management for icommand.

Loads settings from ~/.icommand/config.toml with sensible defaults.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


def get_icommand_dir() -> Path:
    """Return the path to ~/.icommand/, creating it if needed."""
    path = Path.home() / ".icommand"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config_path() -> Path:
    """Return the path to ~/.icommand/config.toml."""
    return get_icommand_dir() / "config.toml"


@dataclass
class Config:
    """Application configuration with sensible defaults."""

    provider: str = "local"
    max_results: int = 10

    # LLM settings for future conversational search
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None


def load_config() -> Config:
    """Load config from ~/.icommand/config.toml, returning defaults if missing."""
    config_path = get_config_path()

    if not config_path.exists():
        return Config()

    data = toml.load(config_path)
    return Config(
        provider=data.get("provider", "local"),
        max_results=data.get("max_results", 10),
        llm_provider=data.get("llm_provider"),
        llm_api_key=data.get("llm_api_key"),
        llm_model=data.get("llm_model"),
    )
