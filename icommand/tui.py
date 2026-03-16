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
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, ListItem, ListView, Static


TUI_FETCH_LIMIT = 100
_FOOTER_BASE = (
    "[#aaaaaa]↑↓[/] nav   "
    "[#aaaaaa]⏎[/] select & copy   "
    "[#aaaaaa]Tab[/] switch focus   "
    "[#aaaaaa]Esc / q[/] quit"
)


class SyncCompleted(Message):
    """Posted when the background sync finishes."""

    def __init__(self, sync_result) -> None:
        super().__init__()
        self.sync_result = sync_result


class ResultsFetched(Message):
    """Posted when a background results fetch finishes."""

    def __init__(
        self,
        generation: int,
        query: str,
        mode: str,
        results: list,
        has_more: bool,
        append: bool,
        desired_visible_limit: int,
        preferred_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.query = query
        self.mode = mode
        self.results = results
        self.has_more = has_more
        self.append = append
        self.desired_visible_limit = desired_visible_limit
        self.preferred_index = preferred_index


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
        width: 1fr;
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
        show_score: bool = True,
    ) -> None:
        super().__init__()
        self.cmd = command
        self.directory = directory
        self.timestamp = timestamp
        self.similarity_score = similarity_score
        self.show_score = show_score

    def compose(self) -> ComposeResult:
        time_str = _relative_time(self.timestamp)
        dir_str = _truncate(self.directory or "~", 30)
        # Truncate command to fit typical terminal width (reserve padding)
        cmd_str = _truncate(self.cmd, 120)

        yield Static(cmd_str, classes="command")
        if self.show_score:
            pct = f"{self.similarity_score * 100:.0f}%"
            bar = _score_bar(self.similarity_score)
            yield Horizontal(
                Static(dir_str, classes="meta"),
                Static(f"{time_str}  {bar} {pct}", classes="score"),
            )
        else:
            yield Horizontal(
                Static(dir_str, classes="meta"),
                Static(time_str, classes="score"),
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


class ResultsListView(ListView):
    """ListView that asks the app for more rows when moving past the end."""

    class LoadMoreRequested(Message):
        """Posted when the user presses down on the last visible row."""

        def __init__(self, list_view: "ResultsListView", index: int) -> None:
            super().__init__()
            self.list_view = list_view
            self.index = index

        @property
        def control(self) -> "ResultsListView":
            return self.list_view

    BINDINGS = [
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("up,k", "cursor_up", "Cursor up", show=False),
        Binding("down,j", "cursor_down", "Cursor down", show=False),
    ]

    def action_cursor_down(self) -> None:
        """Move down, or request more rows if already on the last visible item."""
        if self.index is None:
            if self._nodes:
                self.index = 0
            return

        if self.index >= len(self._nodes) - 1:
            self.post_message(self.LoadMoreRequested(self, self.index))
            return

        super().action_cursor_down()


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
        height: 2;
        padding: 0 2;
        background: #111111;
        border-bottom: solid #333333;
        color: #ffffff;
        text-style: bold;
        content-align: center middle;
    }
    #header-right {
        color: #666666;
        text-align: right;
    }

    /* ── Search bar ──────────────────────────────── */
    #search-bar {
        height: auto;
        padding: 0 2;
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
        width: 1fr;
    }
    ResultItem .score {
        color: #555555;
        text-align: right;
        width: 22;
    }
    Screen > Horizontal {
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
        Binding("escape", "quit", "Quit", show=False, priority=True),
        Binding("q", "quit", "Quit", show=False),
        Binding("up,k", "move_up", "Up", show=False),
        Binding("down,j", "move_down", "Down", show=False),
        Binding("enter", "select", "Select & Copy", show=False),
        Binding("tab", "toggle_focus", "Toggle Focus", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    # Reactive state
    _query: reactive[str] = reactive("", init=False)
    _results: reactive[list] = reactive([], init=False)
    _command_count: reactive[int] = reactive(0, init=False)
    _indexed_count: reactive[int] = reactive(0, init=False)
    _syncing: reactive[bool] = reactive(True, init=False)
    _debounce_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        # Header
        yield Horizontal(
            Static("iCommand", id="header"),
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
            yield ResultsListView(id="results-list")
            yield Static("", id="empty-state")

        # Footer
        yield Static(
            _FOOTER_BASE,
            id="footer",
            markup=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Focus the search input and kick off the initial sync."""
        self._all_results: list = []
        self._visible_limit = 0
        self._has_more_results = False
        self._loading_more = False
        self._closing = False
        self._request_generation = 0
        self._result_mode = "recent"
        self.query_one("#search-input", Input).focus()
        self._start_sync()

    def _start_sync(self) -> None:
        """Sync unembedded commands in a daemon background thread, then load all.

        Uses a daemon thread (not Textual's @work) so that if the user quits
        while sync is in progress, Python won't block on thread.join() at
        interpreter shutdown — the terminal returns instantly.
        """
        import threading

        def _do_sync() -> None:
            from icommand.search import sync

            try:
                result = sync()
            except Exception:
                result = None

            self.post_message(SyncCompleted(result))

        t = threading.Thread(target=_do_sync, daemon=True, name="icommand-sync")
        t.start()

    @on(SyncCompleted)
    def on_sync_completed(self, event: SyncCompleted) -> None:
        """Handle sync completion on the main thread."""
        self._after_sync(event.sync_result)

    def _after_sync(self, sync_result) -> None:
        """Called on the main thread after sync completes."""
        if self._closing or not self.is_mounted:
            return
        self._syncing = False
        self._refresh_count(sync_result)
        if sync_result is not None:
            for message in sync_result.messages[:2]:
                self.notify(message, timeout=3.0)
        # Show initial results (browse mode with empty query)
        self._run_search(self._query)

    def _refresh_count(self, sync_result=None) -> None:
        """Update the header command count display."""
        if self._closing or not self.is_mounted:
            return
        from icommand.db import get_command_count
        from icommand.vector_index import get_vector_index

        try:
            count = (
                sync_result.retained_commands
                if sync_result is not None
                else get_command_count()
            )
        except Exception:
            count = 0

        try:
            indexed_count = (
                sync_result.indexed_commands
                if sync_result is not None
                else get_vector_index().get_indexed_count()
            )
        except Exception:
            indexed_count = 0

        self._command_count = count
        self._indexed_count = indexed_count
        label = (
            "syncing…"
            if self._syncing
            else f"{count} retained · {indexed_count} indexed"
        )
        try:
            self.query_one("#header-right", Static).update(label)
        except NoMatches:
            return

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

    def _run_search(self, query: str) -> None:
        """Start a new paged search or recent-history load."""
        if self._closing or not self.is_mounted:
            return
        generation = self._request_generation + 1
        self._request_generation = generation
        self._result_mode = "search" if query.strip() else "recent"
        self._loading_more = False
        self._has_more_results = False
        self._all_results = []
        self._results = []
        self._visible_limit = self._page_size()
        try:
            self.query_one("#results-list", ListView).index = None
        except NoMatches:
            return
        self._render_results()

        if self._result_mode == "search":
            self._fetch_results(
                query=query,
                generation=generation,
                mode="search",
                limit=min(self._page_size(), TUI_FETCH_LIMIT),
                offset=0,
                append=False,
                desired_visible_limit=self._page_size(),
            )
        else:
            self._fetch_results(
                query=query,
                generation=generation,
                mode="recent",
                limit=min(self._page_size(), TUI_FETCH_LIMIT),
                offset=0,
                append=False,
                desired_visible_limit=self._page_size(),
            )

    @work(thread=True)
    def _fetch_results(
        self,
        query: str,
        generation: int,
        mode: str,
        limit: int,
        offset: int,
        append: bool,
        desired_visible_limit: int,
        preferred_index: Optional[int] = None,
    ) -> None:
        """Fetch one results page in the background and post it to the UI thread."""
        from icommand.db import get_recent_commands
        from icommand.search import SearchResult, search as do_search

        try:
            if mode == "search":
                results = do_search(query, limit)
                has_more = len(results) >= limit and limit < TUI_FETCH_LIMIT
            else:
                rows = get_recent_commands(limit=limit, offset=offset)
                results = [
                    SearchResult(
                        command=row["command"],
                        directory=row.get("directory") or "",
                        timestamp=row["timestamp"],
                        similarity_score=1.0,
                    )
                    for row in rows
                ]
                total_loaded = offset + len(results)
                has_more = len(results) >= limit and total_loaded < TUI_FETCH_LIMIT
        except Exception:
            results = []
            has_more = False

        self.post_message(
            ResultsFetched(
                generation=generation,
                query=query,
                mode=mode,
                results=results,
                has_more=has_more,
                append=append,
                desired_visible_limit=desired_visible_limit,
                preferred_index=preferred_index,
            )
        )

    def _page_size(self) -> int:
        """Return the TUI page size from config."""
        from icommand.config import load_config

        return max(1, load_config().tui_max_results)

    def _can_load_more(self) -> bool:
        """Return whether more rows can be revealed or fetched."""
        return bool(self._results) and (
            self._visible_limit < len(self._all_results) or self._has_more_results
        )

    def _refresh_footer(self) -> None:
        """Update the footer hint based on load-more availability."""
        if self._closing or not self.is_mounted:
            return
        footer = _FOOTER_BASE
        if self._can_load_more():
            footer += "   [#aaaaaa]↓[/] load more"
        try:
            self.query_one("#footer", Static).update(footer)
        except NoMatches:
            return

    def _render_results(self, preferred_index: Optional[int] = None) -> None:
        """Rebuild the visible results list from the loaded result set."""
        if self._closing or not self.is_mounted:
            return
        try:
            list_view = self.query_one("#results-list", ListView)
            empty_state = self.query_one("#empty-state", Static)
        except NoMatches:
            return
        previous_index = list_view.index

        list_view.clear()

        visible_results = self._all_results[: self._visible_limit]
        self._results = visible_results

        if not visible_results:
            empty_label = (
                "No results. Try a different query."
                if self._query.strip()
                else "No commands yet. Run some commands first, then come back!"
            )
            empty_state.update(empty_label)
            empty_state.display = True
            list_view.display = False
            list_view.index = None
            self._refresh_footer()
            return

        empty_state.display = False
        list_view.display = True

        # Hide scores when showing recent commands (empty search)
        show_score = bool(self._query.strip())

        for result in visible_results:
            list_view.append(
                ResultItem(
                    command=result.command,
                    directory=result.directory,
                    timestamp=result.timestamp,
                    similarity_score=result.similarity_score,
                    show_score=show_score,
                )
            )

        target_index: Optional[int] = preferred_index
        if target_index is None and previous_index is not None:
            target_index = previous_index
        if target_index is not None:
            list_view.index = min(target_index, len(visible_results) - 1)

        self._refresh_footer()

    @on(ResultsFetched)
    def on_results_fetched(self, event: ResultsFetched) -> None:
        """Apply fetched results on the main thread."""
        self._apply_fetched_results(event)

    def _apply_fetched_results(self, response: ResultsFetched) -> None:
        """Apply a background fetch response if it still matches the active query."""
        if self._closing or not self.is_mounted:
            return
        if response.generation != self._request_generation:
            return
        if response.query != self._query:
            return
        if response.mode != self._result_mode:
            return

        self._loading_more = False
        if response.append:
            self._all_results.extend(response.results)
        else:
            self._all_results = list(response.results)

        self._visible_limit = min(
            response.desired_visible_limit,
            len(self._all_results),
        )
        self._has_more_results = response.has_more and len(self._all_results) < TUI_FETCH_LIMIT
        self._render_results(preferred_index=response.preferred_index)

    def _request_more_results(self, current_index: int) -> None:
        """Reveal more results, fetching another page if needed."""
        if self._closing or not self.is_mounted:
            return
        if self._loading_more:
            return

        page_size = self._page_size()
        if self._visible_limit < len(self._all_results):
            self._visible_limit = min(
                self._visible_limit + page_size,
                len(self._all_results),
            )
            self._render_results(preferred_index=current_index + 1)
            return

        if not self._has_more_results:
            return

        self._loading_more = True
        desired_visible_limit = min(self._visible_limit + page_size, TUI_FETCH_LIMIT)
        preferred_index = current_index + 1

        if self._result_mode == "search":
            next_limit = min(len(self._all_results) + page_size, TUI_FETCH_LIMIT)
            self._fetch_results(
                query=self._query,
                generation=self._request_generation,
                mode="search",
                limit=next_limit,
                offset=0,
                append=False,
                desired_visible_limit=desired_visible_limit,
                preferred_index=preferred_index,
            )
        else:
            remaining = TUI_FETCH_LIMIT - len(self._all_results)
            next_limit = min(page_size, remaining)
            if next_limit <= 0:
                self._loading_more = False
                self._has_more_results = False
                self._refresh_footer()
                return
            self._fetch_results(
                query=self._query,
                generation=self._request_generation,
                mode="recent",
                limit=next_limit,
                offset=len(self._all_results),
                append=True,
                desired_visible_limit=desired_visible_limit,
                preferred_index=preferred_index,
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_move_up(self) -> None:
        list_view = self.query_one("#results-list", ResultsListView)
        if not list_view.has_focus:
            list_view.focus()
        if list_view.index is None:
            list_view.index = 0
        elif list_view.index > 0:
            list_view.index -= 1

    def action_move_down(self) -> None:
        list_view = self.query_one("#results-list", ResultsListView)
        if not list_view.has_focus:
            list_view.focus()
        if list_view.index is None:
            if self._results:
                list_view.index = 0
        elif self._results and list_view.index < len(self._results) - 1:
            list_view.index += 1
        elif self._results:
            self._request_more_results(list_view.index)

    def action_toggle_focus(self) -> None:
        search = self.query_one("#search-input", Input)
        list_view = self.query_one("#results-list", ResultsListView)
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

    @on(ResultsListView.LoadMoreRequested)
    def on_results_list_view_load_more_requested(
        self, event: ResultsListView.LoadMoreRequested
    ) -> None:
        """Expand the result set when the focused list hits the last visible row."""
        self._request_more_results(event.index)

    def action_quit(self) -> None:
        self._closing = True
        self._request_generation += 1
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self.workers.cancel_all()
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
