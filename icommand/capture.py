"""Command capture for icommand.

Thin wrapper around db.insert_command() for use by the shell hook.
Can be invoked directly: python -m icommand.capture <command> <directory>
"""

import sys
from typing import Optional

from icommand.db import init_db, insert_command


def capture_command(command: str, directory: str, exit_code: Optional[int] = None) -> None:
    """Capture a single command and store it in the database.

    Args:
        command: The shell command that was executed.
        directory: The working directory when the command was run.
        exit_code: The exit code of the command (optional).
    """
    init_db()
    insert_command(command, directory, exit_code)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m icommand.capture <command> <directory>", file=sys.stderr)
        sys.exit(1)

    capture_command(sys.argv[1], sys.argv[2])
