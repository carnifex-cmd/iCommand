"""Interactive TUI for icommand.

Full-screen terminal UI built with Textual. Launches via `icommand tui`.

Design: monochrome — white/gray/black only. Visual hierarchy through
font weight and brightness, not color.

Keybindings:
    Any text      — type to search (input is auto-focused on launch)
    ↑ / ↓         — navigate results
    j / k         — navigate results (vim-style)
    Enter         — copy selected command to clipboard and exit
    Tab           — toggle focus between search and results
    Esc / q       — quit without selecting
    Ctrl+C        — force quit
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, ListItem, ListView, Static


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative_time(timestamp_str: str) -> str:
    """Convert a stored timestamp string to a human-readable relative time."""
    try:
        then = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str

    diff = datetime.now() - then
    seconds = int(diff.total_seconds())

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        m = seconds // 60
        return f"{m}m ago"
    elif seconds < 86400:
        h = seconds // 3600
        return f"{h}h ago"
    elif seconds < 604800:
        d = seconds // 86400
        return f"{d}d ago"
    elif seconds < 2592000:
        w = seconds // 604800
        return f"{w}w ago"
    else:
        mo = seconds // 2592000
        return f"{mo}mo ago"


def _score_bar(score: float, width: int = 8) -> str:
    """Render a compact ASCII bar for a similarity score (0-1)."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with an ellipsis if it exceeds max_len."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns True on success."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class ResultItem(ListItem):
    """A single search result row in the results list.

    Displays:
        - Command text (bold, prominent)
        - Directory | relative time | score bar + percentage
    """

    DEFAULT_CSS = """
    ResultItem {
        height: 2;
        padding: 0 1;
        margin: 0;
        border-bottom: solid #222222;
    }
    ResultItem:focus-within {
        background: $surface-lighten-1;
    }
    ResultItem.--highlight {
        background: $surface-lighten-1;
    }
    ResultItem .command {
        text-style: bold;
        color: $text;
        height: 1;
        padding: 0;
        margin: 0;
    }
    ResultItem .meta {
        color: $text-muted;
        height: 1;
        padding: 0;
        margin: 0;
    }
    ResultItem .score {
        color: $text-disabled;
        text-align: right;
    }
    ResultItem Horizontal {
        height: 1;
        padding: 0;
        margin: 0;
    }
    """

    def __init__(
        self,
        command: str,
        directory: str,
        timestamp: str,
        similarity_score: float,
    ) -> None:
        super().__init__()
        self.cmd = command
        self.directory = directory
        self.timestamp = timestamp
        self.similarity_score = similarity_score

    def compose(self) -> ComposeResult:
        pct = f"{self.similarity_score * 100:.0f}%"
        bar = _score_bar(self.similarity_score)
        time_str = _relative_time(self.timestamp)
        dir_str = _truncate(self.directory or "~", 40)

        yield Static(self.cmd, classes="command")
        yield Horizontal(
            Static(dir_str, classes="meta"),
            Static(f"{time_str}  {bar} {pct}", classes="score"),
        )


class EmptyState(Static):
    """Shown when there are no results to display."""

    DEFAULT_CSS = """
    EmptyState {
        content-align: center middle;
        height: 100%;
        color: $text-disabled;
    }
    """


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class ICommandApp(App):
    """iCommand TUI — monochrome, semantic history search."""

    TITLE = "iCommand"
    SUB_TITLE = "Semantic History Search"

    # Monochrome theme: leverage textual's built-in dark palette,
    # override only specific variables for a cleaner, flatter look.
    CSS = """
    Screen {
        background: #111111;
    }

    /* ── Header ──────────────────────────────────── */
    #header {
        height: 1;
        padding: 0 2;
        background: #111111;
        border-bottom: solid #333333;
        color: #ffffff;
        text-style: bold;
    }
    #header-right {
        color: #666666;
        text-align: right;
    }

    /* ── Search bar ──────────────────────────────── */
    #search-bar {
        height: auto;
        padding: 1 2;
        background: #111111;
        border-bottom: solid #333333;
    }
    #search-input {
        background: #1e1e1e;
        border: solid #444444;
        color: #ffffff;
        padding: 0 1;
    }
    #search-input:focus {
        border: solid #aaaaaa;
    }

    /* ── Results ─────────────────────────────────── */
    #results-container {
        height: 1fr;
        background: #111111;
    }
    #results-list {
        background: #111111;
        border: none;
        scrollbar-color: #333333;
        scrollbar-background: #111111;
        overflow-y: auto;
        padding: 0;
    }
    #results-list > ListItem {
        padding: 0 1;
        margin: 0;
        height: 2;
    }
    #results-list > ListItem.--highlight {
        background: #1e1e1e;
    }
    ResultItem .command {
        color: #ffffff;
        text-style: bold;
    }
    ResultItem .meta {
        color: #666666;
    }
    ResultItem .score {
        color: #555555;
        text-align: right;
        width: 1fr;
    }
    Horizontal {
        height: auto;
    }

    /* ── Empty state ─────────────────────────────── */
    #empty-state {
        height: 100%;
        background: #111111;
        color: #444444;
        content-align: center middle;
    }

    /* ── Footer ──────────────────────────────────── */
    #footer {
        height: 2;
        padding: 0 2;
        background: #111111;
        border-top: solid #333333;
        color: #666666;
        content-align: left middle;
    }
    #footer .key {
        color: #ffffff;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("up,k", "move_up", "Up", show=False),
        Binding("down,j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select & Copy", show=False),
        Binding("tab", "toggle_focus", "Toggle Focus", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    # Reactive state
    _query: reactive[str] = reactive("", init=False)
    _results: reactive[list] = reactive([], init=False)
    _command_count: reactive[int] = reactive(0, init=False)
    _syncing: reactive[bool] = reactive(True, init=False)
    _debounce_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        # Header
        yield Horizontal(
            Static("⚡  iCommand", id="header"),
            Static("", id="header-right"),
        )

        # Search input
        with Vertical(id="search-bar"):
            yield Input(
                placeholder="Type to search your command history…",
                id="search-input",
            )

        # Results area
        with Vertical(id="results-container"):
            yield ListView(id="results-list")
            yield Static("", id="empty-state")

        # Footer
        yield Static(
            "[#aaaaaa]↑↓[/] nav   "
            "[#aaaaaa]⏎[/] select & copy   "
            "[#aaaaaa]Tab[/] switch focus   "
            "[#aaaaaa]Esc / q[/] quit",
            id="footer",
            markup=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Focus the search input and kick off the initial sync."""
        self.query_one("#search-input", Input).focus()
        self._start_sync()

    @work(thread=True)
    def _start_sync(self) -> None:
        """Sync unembedded commands in a background thread, then load all."""
        from icommand.search import sync

        try:
            synced = sync()
        except Exception:
            synced = 0

        self.call_from_thread(self._after_sync, synced)

    def _after_sync(self, synced: int) -> None:
        """Called on the main thread after sync completes."""
        self._syncing = False
        self._refresh_count()
        # Show initial results (browse mode with empty query)
        self._run_search(self._query)

    def _refresh_count(self) -> None:
        """Update the header command count display."""
        from icommand.db import get_all_commands

        try:
            count = len(get_all_commands())
        except Exception:
            count = 0

        self._command_count = count
        label = "syncing…" if self._syncing else f"{count} commands indexed"
        self.query_one("#header-right", Static).update(label)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @on(Input.Changed, "#search-input")
    def on_search_input_changed(self, event: Input.Changed) -> None:
        """Debounce search queries to 300 ms."""
        self._query = event.value

        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        self._debounce_task = asyncio.create_task(
            self._debounced_search(event.value)
        )

    async def _debounced_search(self, query: str) -> None:
        await asyncio.sleep(0.3)
        self._run_search(query)

    @work(thread=True)
    def _run_search(self, query: str) -> None:
        """Run semantic search in a background thread."""
        from icommand.search import search as do_search

        try:
            if query.strip():
                from icommand.config import load_config
                config = load_config()
                results = do_search(query, config.max_results)
            else:
                # Empty query: show most recent commands
                results = self._recent_commands()
        except Exception:
            results = []

        self.call_from_thread(self._update_results, results)

    def _recent_commands(self) -> list:
        """Return the most recent commands as SearchResult-like objects."""
        from icommand.db import get_all_commands
        from icommand.search import SearchResult

        try:
            rows = get_all_commands()
            return [
                SearchResult(
                    command=r["command"],
                    directory=r.get("directory") or "",
                    timestamp=r["timestamp"],
                    similarity_score=1.0,
                )
                for r in rows[:20]
            ]
        except Exception:
            return []

    def _update_results(self, results: list) -> None:
        """Rebuild the results list widget on the main thread."""
        from icommand.config import load_config
        
        # Limit results to tui_max_results setting
        config = load_config()
        results = results[:config.tui_max_results]
        
        self._results = results
        list_view = self.query_one("#results-list", ListView)
        empty_state = self.query_one("#empty-state", Static)

        list_view.clear()

        if not results:
            empty_label = (
                "No results. Try a different query."
                if self._query.strip()
                else "No commands yet. Run some commands first, then come back!"
            )
            empty_state.update(empty_label)
            empty_state.display = True
            list_view.display = False
            return

        empty_state.display = False
        list_view.display = True

        for result in results:
            list_view.append(
                ResultItem(
                    command=result.command,
                    directory=result.directory,
                    timestamp=result.timestamp,
                    similarity_score=result.similarity_score,
                )
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_move_up(self) -> None:
        list_view = self.query_one("#results-list", ListView)
        if not list_view.has_focus:
            list_view.focus()
        if list_view.index is None:
            list_view.index = 0
        elif list_view.index > 0:
            list_view.index -= 1

    def action_move_down(self) -> None:
        list_view = self.query_one("#results-list", ListView)
        if not list_view.has_focus:
            list_view.focus()
        if list_view.index is None:
            list_view.index = 0
        elif self._results and list_view.index < len(self._results) - 1:
            list_view.index += 1

    def action_toggle_focus(self) -> None:
        search = self.query_one("#search-input", Input)
        list_view = self.query_one("#results-list", ListView)
        if search.has_focus:
            list_view.focus()
        else:
            search.focus()

    def action_select(self) -> None:
        """Copy the highlighted command to clipboard and exit."""
        list_view = self.query_one("#results-list", ListView)
        idx = list_view.index

        if idx is None or not self._results:
            return
        if idx >= len(self._results):
            return

        command = self._results[idx].command
        copied = _copy_to_clipboard(command)

        # Print the command to stdout so shell wrapper scripts can capture it
        import sys
        print(command, file=sys.stdout)

        if copied:
            self.notify(f"Copied: {command}", timeout=1.5)

        self.exit(command)

    def action_quit(self) -> None:
        self.exit()

    # ------------------------------------------------------------------
    # Enter key on ListView selects
    # ------------------------------------------------------------------

    @on(ListView.Selected)
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.action_select()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch() -> None:
    """Launch the iCommand TUI. Called by the CLI `icommand tui` command."""
    app = ICommandApp()
    result = app.run()
    # If a command was selected (exit with result), it was already printed
    # to stdout by action_select so shell wrappers can capture it.
