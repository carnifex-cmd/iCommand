"""CLI entry point for icommand.

Provides commands: init, search, ask, uninstall, capture.
"""

import shutil
from datetime import datetime
from pathlib import Path

import click

from icommand.config import get_config_path, get_icommand_dir, load_config, save_config, Config
from icommand.db import init_db
from icommand.search import search as do_search
from icommand.search import sync


def _relative_time(timestamp_str: str) -> str:
    """Convert a timestamp string to a human-readable relative time."""
    try:
        then = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str

    now = datetime.now()
    diff = now - then

    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < 2592000:
        weeks = seconds // 604800
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    else:
        months = seconds // 2592000
        return f"{months} month{'s' if months != 1 else ''} ago"


def _get_hook_path() -> Path:
    """Return the absolute path to hook.sh bundled inside the package."""
    return Path(__file__).parent / "hook.sh"


def _get_hook_source_line() -> str:
    """Return the line that should be appended to shell rc files."""
    return f'\n# icommand: AI-powered command history search\nsource "{_get_hook_path()}"\n'


_HOOK_MARKER = "# icommand: AI-powered command history search"


@click.group()
def cli():
    """icommand — AI-powered terminal command history search."""
    pass


@cli.command(hidden=True)
@click.argument("cmd")
@click.argument("directory")
def capture(cmd: str, directory: str):
    """Record a command (called by the shell hook — not for direct use)."""
    from icommand.capture import capture_command
    capture_command(cmd, directory)


@cli.command()
def init():
    """Initialize icommand: set up database, config, and shell hook."""
    icommand_dir = get_icommand_dir()
    click.echo(f"  created  {icommand_dir}")

    # Initialize the database
    init_db()
    click.echo("  db       initialized")

    # Write default config if it doesn't exist
    config_path = get_config_path()
    if not config_path.exists():
        example_config = Path(__file__).parent.parent / "config.example.toml"
        if example_config.exists():
            shutil.copy(example_config, config_path)
        else:
            config_path.write_text(
                '# icommand configuration\nprovider = "local"\nmax_results = 10\n'
            )
        click.echo(f"  config   {config_path}")
    else:
        click.echo(f"  config   {config_path}")

    # Append hook to shell rc files
    hook_line = _get_hook_source_line()
    hook_path = _get_hook_path()

    if not hook_path.exists():
        click.echo(f"  warning  hook not found at {hook_path}", err=True)
    else:
        for rc_name in [".bashrc", ".zshrc"]:
            rc_path = Path.home() / rc_name
            if rc_path.exists():
                content = rc_path.read_text()
                if _HOOK_MARKER not in content:
                    with open(rc_path, "a") as f:
                        f.write(hook_line)
                    click.echo(f"  hook     appended to ~/{rc_name}")
                else:
                    click.echo(f"  hook     ~/{rc_name} (already present)")

    click.echo()
    click.echo("  ready.")
    click.echo()
    click.echo("  open a new terminal, run some commands, then:")
    click.echo('    icommand search "<query>"')


@cli.command()
@click.argument("query")
def search(query: str):
    """Semantically search your command history."""
    config = load_config()

    click.echo("syncing...", nl=False)
    synced = sync()
    if synced > 0:
        click.echo(f" {synced} command{'s' if synced != 1 else ''} synced.")
    else:
        click.echo(" up to date.")

    click.echo()

    results = do_search(query, config.max_results)

    if not results:
        click.echo("No matching commands found.")
        click.echo("Tip: Make sure you have some command history captured first.")
        return

    click.echo(f"{len(results)} result{'s' if len(results) != 1 else ''}:\n")

    for i, result in enumerate(results, 1):
        time_ago = _relative_time(result.timestamp)
        similarity_pct = f"{result.similarity_score * 100:.0f}%"

        click.echo(f"  {i}. {click.style(result.command, fg='green', bold=True)}")
        click.echo(f"     in  {result.directory or 'unknown'}")
        click.echo(f"     at  {time_ago}  ·  {similarity_pct} match")
        click.echo()


@cli.command()
@click.argument("question")
def ask(question: str):
    """Ask a natural language question about your command history."""
    click.echo("conversational search is not yet available.")
    click.echo()
    click.echo("use: icommand search \"<query>\"")


@cli.command()
def uninstall():
    """Remove icommand data and shell hooks from your system."""
    icommand_dir = get_icommand_dir()

    if not click.confirm(
        f"This will delete {icommand_dir} and remove shell hooks. Continue?"
    ):
        click.echo("Cancelled.")
        return

    # Remove hook lines from shell rc files
    for rc_name in [".bashrc", ".zshrc"]:
        rc_path = Path.home() / rc_name
        if rc_path.exists():
            lines = rc_path.read_text().splitlines(keepends=True)
            cleaned = []
            skip_next = False
            for line in lines:
                if _HOOK_MARKER in line:
                    skip_next = True
                    continue
                if skip_next and line.strip().startswith("source") and "hook.sh" in line:
                    skip_next = False
                    continue
                skip_next = False
                cleaned.append(line)
            rc_path.write_text("".join(cleaned))
            click.echo(f"  hook     removed from ~/{rc_name}")

    # Remove the ~/.icommand directory
    if icommand_dir.exists():
        shutil.rmtree(icommand_dir)
        click.echo(f"  removed  {icommand_dir}")

    click.echo("  done.")
    click.echo()
    click.echo("  to finish, run:")
    click.echo("    pip uninstall icommand")


@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--reset", is_flag=True, help="Reset all settings to defaults")
def config(key, value, reset):
    """View or update configuration settings.
    
    Show current config:
        icommand config
    
    Update a setting:
        icommand config max_results 5
    
    Reset to defaults:
        icommand config --reset
    
    Available settings:
        max_results     Maximum number of search results (default: 10)
        provider        Embedding provider: local, openai, anthropic, ollama (default: local)
    """
    config_path = get_config_path()
    
    if reset:
        save_config(Config())
        click.echo("  config   reset to defaults")
        return
    
    # Show current config
    if key is None:
        cfg = load_config()
        click.echo(f"Configuration file: {config_path}")
        click.echo()
        click.echo(f"  max_results = {cfg.max_results}")
        click.echo(f"  provider    = {cfg.provider}")
        if cfg.llm_provider:
            click.echo(f"  llm_provider = {cfg.llm_provider}")
        if cfg.llm_model:
            click.echo(f"  llm_model    = {cfg.llm_model}")
        return
    
    # Validate key
    valid_keys = ["max_results", "provider"]
    if key not in valid_keys:
        click.echo(f"Error: Unknown setting '{key}'", err=True)
        click.echo(f"Valid settings: {', '.join(valid_keys)}", err=True)
        return
    
    # Get current config and update
    cfg = load_config()
    
    if value is None:
        # Show specific key
        click.echo(f"  {key} = {getattr(cfg, key)}")
        return
    
    # Update the value
    if key == "max_results":
        try:
            value = int(value)
            if value < 1 or value > 100:
                click.echo("Error: max_results must be between 1 and 100", err=True)
                return
        except ValueError:
            click.echo("Error: max_results must be a number", err=True)
            return
    
    setattr(cfg, key, value)
    save_config(cfg)
    click.echo(f"  config   {key} = {value}")
