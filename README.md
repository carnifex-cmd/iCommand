# iCommand

`iCommand` is a local-first command history search tool for people who spend a lot of time in the terminal and want better recall than shell history alone.

It captures commands from your shell, stores them locally, and lets you search them through a full-screen TUI or a quick CLI command. Literal keyword lookups are supported, and recent history can also be searched semantically using local embeddings.

## Why It Exists

Traditional shell history is fast, but it is not great at answering questions like:

- "What was that `pipx` command I ran last week?"
- "How did I activate that virtualenv?"
- "What did I use to reinstall this tool?"

`iCommand` is built to make those lookups easier without turning your history into a cloud product.

## Current Status

This project is usable today, but it is still early.

- Local embeddings are implemented with Snowflake Arctic Embed XS via ONNX.
- The TUI and CLI search flows are implemented.
- Keyword search is backed by SQLite FTS.
- Local storage is bounded so the app does not grow forever.
- `ask` / conversational search is not implemented yet.
- Only the `local` embedding provider is implemented right now. `openai`, `anthropic`, and `ollama` are present as future placeholders.

## Features

- Full-screen terminal UI for browsing and searching command history
- CLI search for quick one-off lookups
- Exact and prefix keyword matching for literal shell queries
- Semantic search for recent history using local embeddings
- Shell history import for existing Bash and Zsh users
- Automatic local storage limits with pruning of oldest history
- Semantic indexing limited to a recent hot window to keep disk usage bounded

## Installation

### Prerequisites

- Python `3.9+`
- `pipx`
- Bash or Zsh
- Internet access on first semantic-search use to download the ONNX embedding model from Hugging Face

### Install From GitHub

```bash
pipx install 'git+https://github.com/carnifex-cmd/iCommand.git'
```

Initialize the app and install the shell hook:

```bash
icommand init
```

Then open a new terminal, or reload your shell config.

## Quick Start

Open the TUI:

```bash
ic
```

Run a quick CLI search:

```bash
icommand search "reinstall"
```

Import recent shell history:

```bash
icommand import-history
```

Inspect current settings:

```bash
icommand config
```

Uninstall the app:

```bash
icommand uninstall
```

## Commands

### `icommand init`

Initializes the local database, writes a default config file, appends the shell hook to `~/.bashrc` and `~/.zshrc`, and performs an initial sync so search works immediately.

### `icommand tui`

Launches the full-screen Textual TUI.

The shell hook also defines:

```bash
alias ic='icommand tui'
```

### `icommand search "<query>"`

Runs a CLI search against your local history.

### `icommand import-history`

Imports commands from `~/.zsh_history` or `~/.bash_history`.

Useful options:

```bash
icommand import-history --limit 5000
icommand import-history --file ~/.zsh_history
```

### `icommand config`

Shows or updates config values.

Examples:

```bash
icommand config
icommand config max_results 10
icommand config tui_max_results 5
icommand config storage_soft_limit_mb 1024
```

### `icommand uninstall`

Removes local app data, shell hook lines, and attempts to uninstall the binary.

## How It Works

1. A shell hook captures commands from Bash or Zsh.
2. Commands are stored in a local SQLite database under `~/.icommand/history.db`.
3. SQLite FTS is used for keyword search.
4. Recent commands can also be embedded locally and indexed in FAISS for semantic search.
5. A storage maintenance pass keeps disk growth bounded by pruning the oldest history and limiting semantic indexing to a recent hot window.

## Configuration

Default config lives at `~/.icommand/config.toml`.

Important settings:

```toml
provider = "local"
max_results = 10
tui_max_results = 5
storage_soft_limit_mb = 1024
storage_hard_limit_mb = 2048
live_command_limit = 1000000
semantic_command_limit = 250000
```

What they mean:

- `provider`: embedding backend. Only `local` works today.
- `max_results`: maximum results returned by CLI search.
- `tui_max_results`: maximum visible results in the TUI.
- `storage_soft_limit_mb`: target cap for local app data under `~/.icommand`.
- `storage_hard_limit_mb`: emergency cap after which new embedding work pauses until storage is reduced.
- `live_command_limit`: total retained commands kept locally.
- `semantic_command_limit`: number of newest commands eligible for semantic indexing.

## Privacy and Local Data

By default, `iCommand` is local-first.

It stores:

- command text
- working directory
- timestamp
- exit code when available
- local embeddings for semantically indexed commands
- a local FAISS index and metadata

Primary storage locations:

- `~/.icommand/history.db`
- `~/.icommand/vectors.faiss`
- `~/.icommand/vectors_metadata.pkl`

The local embedding model is downloaded and cached separately by Hugging Face tooling, typically under:

- `~/.cache/huggingface/hub`

## Development

Clone the repo and install it locally with `pipx`:

```bash
git clone https://github.com/carnifex-cmd/iCommand.git
cd iCommand
pipx install --force --editable .
```

Useful local commands:

```bash
icommand init
icommand config
ic
```

If you are working from source and want the installed binary to use your local edits, prefer the editable `pipx` install above instead of reinstalling directly from the GitHub URL.

## Contributing

Issues and pull requests are welcome.

When contributing:

- keep behavior local-first
- avoid unbounded disk or memory growth
- preserve fast literal shell lookup behavior
- document user-visible limitations honestly

## Roadmap

Likely next areas of work:

- better live-search freshness while the TUI stays open
- finishing additional embedding providers
- conversational command recall
- stronger uninstall cleanup
- tests and release automation

## License

Intended license: `MIT`.

If you publish this repository, add a matching `LICENSE` file so the legal terms are explicit in the repo itself.
