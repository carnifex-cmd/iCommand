#!/usr/bin/env bash
# icommand shell hook — captures every command for bash and zsh
# Source this file from your .bashrc or .zshrc:
#   source /path/to/hook.sh

# Track state between preexec and precmd
_ICOMMAND_LAST_CMD=""
_ICOMMAND_PENDING_CMD=""
_ICOMMAND_CMD_START=""
_ICOMMAND_LAST_HISTNO=""

_icommand_capture() {
    local cmd="$1"

    # Skip empty commands
    if [ -z "$cmd" ]; then
        return
    fi

    # Skip consecutive duplicates
    if [ "$cmd" = "$_ICOMMAND_LAST_CMD" ]; then
        return
    fi

    _ICOMMAND_LAST_CMD="$cmd"

    # Call the globally-installed icommand CLI in the background.
    # Run in a subshell so zsh does not track it as a named job
    # (prevents "[N] + done ..." notifications in the prompt).
    local exit_code="${2:-}"
    if [ -n "$exit_code" ]; then
        (icommand capture "$cmd" "$PWD" --exit-code "$exit_code" 2>/dev/null &)
    else
        (icommand capture "$cmd" "$PWD" 2>/dev/null &)
    fi
}

# --- Bash hook ---
if [ -n "$BASH_VERSION" ]; then
    _icommand_seed_bash_history_state() {
        local last_history_line histno
        last_history_line=$(HISTTIMEFORMAT='' history 1 2>/dev/null)
        histno=$(printf '%s\n' "$last_history_line" | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*/\1/p')
        _ICOMMAND_LAST_HISTNO="$histno"
    }

    _icommand_prompt_command() {
        local last_history_line histno last_cmd
        last_history_line=$(HISTTIMEFORMAT='' history 1 2>/dev/null)
        histno=$(printf '%s\n' "$last_history_line" | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*/\1/p')
        if [ -z "$histno" ]; then
            return
        fi
        if [ "$histno" = "$_ICOMMAND_LAST_HISTNO" ]; then
            return
        fi
        _ICOMMAND_LAST_HISTNO="$histno"
        last_cmd=$(printf '%s\n' "$last_history_line" | sed 's/^[ ]*[0-9]*[ ]*//')
        _icommand_capture "$last_cmd"
    }

    _icommand_seed_bash_history_state

    # Append to PROMPT_COMMAND so we don't clobber existing hooks
    if [ -z "$PROMPT_COMMAND" ]; then
        PROMPT_COMMAND="_icommand_prompt_command"
    else
        PROMPT_COMMAND="_icommand_prompt_command; $PROMPT_COMMAND"
    fi
fi

# --- Zsh hook ---
if [ -n "$ZSH_VERSION" ]; then

    # preexec: fires just BEFORE a command runs — receives the command as $1
    _icommand_preexec() {
        _ICOMMAND_PENDING_CMD="$1"
    }

    # precmd: fires just AFTER a command finishes, BEFORE the next prompt
    _icommand_precmd() {
        local exit_code=$?
        local cmd="$_ICOMMAND_PENDING_CMD"

        _ICOMMAND_PENDING_CMD=""

        # Only capture real commands that passed through preexec.
        # Do not fall back to shell history here, or shell startup / prompt redraws
        # can replay stale commands as if they were just executed.
        if [ -z "$cmd" ]; then
            return
        fi

        _icommand_capture "$cmd" "$exit_code"
    }

    # Register hooks if not already present
    if [[ -z "${preexec_functions[(r)_icommand_preexec]}" ]]; then
        preexec_functions+=(_icommand_preexec)
    fi
    if [[ -z "${precmd_functions[(r)_icommand_precmd]}" ]]; then
        precmd_functions+=(_icommand_precmd)
    fi
fi

# ic — shortcut to open the iCommand TUI
alias ic='icommand tui'
