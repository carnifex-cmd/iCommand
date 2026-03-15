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
    tui_max_results: int = 5  # Number of results shown in TUI
    storage_soft_limit_mb: int = 1024
    storage_hard_limit_mb: int = 2048
    live_command_limit: int = 1_000_000
    semantic_command_limit: int = 250_000

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
        tui_max_results=data.get("tui_max_results", 5),
        storage_soft_limit_mb=data.get("storage_soft_limit_mb", 1024),
        storage_hard_limit_mb=data.get("storage_hard_limit_mb", 2048),
        live_command_limit=data.get("live_command_limit", 1_000_000),
        semantic_command_limit=data.get("semantic_command_limit", 250_000),
        llm_provider=data.get("llm_provider"),
        llm_api_key=data.get("llm_api_key"),
        llm_model=data.get("llm_model"),
    )


def save_config(config: Config) -> None:
    """Save config to ~/.icommand/config.toml."""
    config_path = get_config_path()
    data = {
        "provider": config.provider,
        "max_results": config.max_results,
        "tui_max_results": config.tui_max_results,
        "storage_soft_limit_mb": config.storage_soft_limit_mb,
        "storage_hard_limit_mb": config.storage_hard_limit_mb,
        "live_command_limit": config.live_command_limit,
        "semantic_command_limit": config.semantic_command_limit,
    }
    # Only include LLM settings if they're set
    if config.llm_provider:
        data["llm_provider"] = config.llm_provider
    if config.llm_api_key:
        data["llm_api_key"] = config.llm_api_key
    if config.llm_model:
        data["llm_model"] = config.llm_model

    with open(config_path, "w") as f:
        toml.dump(data, f)
