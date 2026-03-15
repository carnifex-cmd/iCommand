"""Storage maintenance for icommand's bounded local history."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from icommand.config import Config, get_icommand_dir
from icommand.db import (
    clear_embeddings_before_id,
    delete_oldest_commands,
    ensure_incremental_auto_vacuum,
    get_app_state,
    get_command_count,
    get_db_page_stats,
    get_max_command_id,
    optimize_database,
    set_app_state,
    vacuum_database,
)
from icommand.vector_index import CURRENT_MODEL, INDEX_FILE, METADATA_FILE, get_vector_index

MB = 1024 * 1024
GB = 1024 * MB
LOW_DISK_THRESHOLD_BYTES = 20 * GB
SOFT_LIMIT_FLOOR_BYTES = 256 * MB
HARD_LIMIT_FLOOR_BYTES = 512 * MB
PRUNE_BATCH_SIZE = 10_000
FULL_VACUUM_TRIGGER_ROWS = 50_000
FREELIST_VACUUM_THRESHOLD_BYTES = 64 * MB
PRUNE_NOTICE_KEY = "prune_notice_shown"


@dataclass
class StorageLimits:
    soft_limit_bytes: int
    hard_limit_bytes: int
    free_bytes: int


@dataclass
class MaintenanceReport:
    soft_limit_bytes: int
    hard_limit_bytes: int
    pruned_rows: int = 0
    pruned_for_count: int = 0
    pruned_for_storage: int = 0
    cold_embeddings_cleared: int = 0
    retained_commands: int = 0
    hot_floor_id: int = 1
    current_usage_bytes: int = 0
    embedding_paused: bool = False
    rebuild_required: bool = False
    show_prune_notice: bool = False
    messages: list[str] = field(default_factory=list)


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def get_storage_usage_bytes() -> int:
    """Return total on-disk usage for icommand's own state."""
    icommand_dir = get_icommand_dir()
    return sum(
        _file_size(icommand_dir / name)
        for name in ("history.db", INDEX_FILE, METADATA_FILE)
    )


def get_hot_floor_id(max_command_id: int, semantic_command_limit: int) -> int:
    """Return the first command ID that belongs to the semantic hot window."""
    if max_command_id <= 0:
        return 1
    limit = max(1, semantic_command_limit)
    return max(1, max_command_id - limit + 1)


def get_effective_storage_limits(config: Config) -> StorageLimits:
    """Compute soft/hard storage caps with low-disk adaptation."""
    icommand_dir = get_icommand_dir()
    disk = shutil.disk_usage(icommand_dir)

    soft_limit = max(SOFT_LIMIT_FLOOR_BYTES, config.storage_soft_limit_mb * MB)
    hard_limit = max(HARD_LIMIT_FLOOR_BYTES, config.storage_hard_limit_mb * MB)
    if hard_limit < soft_limit:
        hard_limit = soft_limit

    if disk.free < LOW_DISK_THRESHOLD_BYTES:
        soft_limit = max(
            SOFT_LIMIT_FLOOR_BYTES,
            min(soft_limit, int(disk.free * 0.05)),
        )
        hard_limit = max(
            HARD_LIMIT_FLOOR_BYTES,
            min(hard_limit, int(disk.free * 0.10)),
        )
        if hard_limit < soft_limit:
            hard_limit = soft_limit

    return StorageLimits(
        soft_limit_bytes=soft_limit,
        hard_limit_bytes=hard_limit,
        free_bytes=disk.free,
    )


def _reclaim_database_space(force_full: bool = False) -> None:
    db_path = get_icommand_dir() / "history.db"
    before_size = _file_size(db_path)
    optimize_database()
    stats = get_db_page_stats()
    freelist_bytes = stats["page_size"] * stats["freelist_count"]
    after_incremental = _file_size(db_path)

    if force_full or (
        freelist_bytes >= FREELIST_VACUUM_THRESHOLD_BYTES
        and before_size - after_incremental < max(4 * MB, freelist_bytes // 4)
    ):
        vacuum_database()


def run_maintenance(config: Config, *, model_name: str = CURRENT_MODEL) -> MaintenanceReport:
    """Apply storage limits, enforce the hot window, and invalidate stale indexes."""
    ensure_incremental_auto_vacuum()

    limits = get_effective_storage_limits(config)
    report = MaintenanceReport(
        soft_limit_bytes=limits.soft_limit_bytes,
        hard_limit_bytes=limits.hard_limit_bytes,
    )

    retained_count = get_command_count()
    usage_bytes = get_storage_usage_bytes()
    size_prune_full_vacuum_done = False

    while retained_count > config.live_command_limit:
        deleted = delete_oldest_commands(
            min(PRUNE_BATCH_SIZE, retained_count - config.live_command_limit)
        )
        if deleted <= 0:
            break
        report.pruned_rows += deleted
        report.pruned_for_count += deleted
        retained_count -= deleted

    if report.pruned_rows > 0:
        _reclaim_database_space(force_full=report.pruned_rows >= FULL_VACUUM_TRIGGER_ROWS)
        usage_bytes = get_storage_usage_bytes()

    while usage_bytes > limits.soft_limit_bytes and retained_count > 0:
        deleted = delete_oldest_commands(PRUNE_BATCH_SIZE)
        if deleted <= 0:
            break
        report.pruned_rows += deleted
        report.pruned_for_storage += deleted
        retained_count -= deleted
        force_full = (
            report.pruned_for_storage >= FULL_VACUUM_TRIGGER_ROWS
            and not size_prune_full_vacuum_done
        )
        _reclaim_database_space(force_full=force_full)
        if force_full:
            size_prune_full_vacuum_done = True
        usage_bytes = get_storage_usage_bytes()

    max_command_id = get_max_command_id()
    hot_floor_id = get_hot_floor_id(max_command_id, config.semantic_command_limit)
    report.hot_floor_id = hot_floor_id

    metadata = get_vector_index().load_metadata()
    current_hot_floor = int(metadata.get("hot_floor_id", 1))
    model_mismatch = metadata.get("model") != model_name
    cold_embeddings_cleared = clear_embeddings_before_id(hot_floor_id)
    report.cold_embeddings_cleared = cold_embeddings_cleared

    if (
        report.pruned_rows > 0
        or cold_embeddings_cleared > 0
        or metadata.get("rebuild_needed", False)
        or current_hot_floor != hot_floor_id
        or model_mismatch
    ):
        get_vector_index().invalidate(
            hot_floor_id=hot_floor_id,
            model=model_name,
            delete_index=True,
        )
        report.rebuild_required = True

    if cold_embeddings_cleared > 0:
        _reclaim_database_space(
            force_full=(
                report.pruned_rows >= FULL_VACUUM_TRIGGER_ROWS
                or cold_embeddings_cleared >= FULL_VACUUM_TRIGGER_ROWS
            )
        )

    report.current_usage_bytes = get_storage_usage_bytes()
    report.retained_commands = get_command_count()
    report.embedding_paused = report.current_usage_bytes > limits.hard_limit_bytes

    if report.pruned_rows > 0:
        if get_app_state(PRUNE_NOTICE_KEY) != "1":
            set_app_state(PRUNE_NOTICE_KEY, "1")
            report.show_prune_notice = True
        report.messages.append(
            f"pruned {report.pruned_rows} old commands to enforce the local storage budget"
        )

    if report.embedding_paused:
        report.messages.append(
            "storage limit reached; semantic embedding is paused until local data shrinks"
        )

    return report
