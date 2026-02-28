#!/usr/bin/env bash
# icommand shell hook — captures every command for bash and zsh
# Source this file from your .bashrc or .zshrc:
#   source /path/to/hook.sh

# Track the last captured command to skip consecutive duplicates
_ICOMMAND_LAST_CMD=""

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

    # Call the globally-installed icommand CLI in the background
    # This works regardless of which Python version is active
    icommand capture "$cmd" "$PWD" 2>/dev/null &
}

# --- Bash hook ---
if [ -n "$BASH_VERSION" ]; then
    _icommand_prompt_command() {
        local last_cmd
        last_cmd=$(HISTTIMEFORMAT='' history 1 | sed 's/^[ ]*[0-9]*[ ]*//')
        _icommand_capture "$last_cmd"
    }

    # Append to PROMPT_COMMAND so we don't clobber existing hooks
    if [ -z "$PROMPT_COMMAND" ]; then
        PROMPT_COMMAND="_icommand_prompt_command"
    else
        PROMPT_COMMAND="_icommand_prompt_command; $PROMPT_COMMAND"
    fi
fi

# --- Zsh hook ---
if [ -n "$ZSH_VERSION" ]; then
    _icommand_precmd() {
        local last_cmd
        last_cmd=$(fc -ln -1 2>/dev/null | sed 's/^[ ]*//')
        _icommand_capture "$last_cmd"
    }

    # Add to precmd_functions array if not already there
    if (( ${precmd_functions[(I)_icommand_precmd]} == 0 )); then
        precmd_functions+=(_icommand_precmd)
    fi
fi
