# Agent Stack

This is the neutral, user-level coordination layer for terminal agents on this host.

## Layout

- `~/.config/agent-stack/`: declarative configuration shared conceptually by clients.
- `~/.agents/skills/`: canonical reusable skills.
- `~/.agents/knowledge/`: stable cross-project knowledge; not client runtime memory.
- `~/.local/share/agent-square/`: generated conversation index and runtime state.
- `~/.local/bin/agent-square`: SSH-friendly thread browser and launcher.

Tool-owned databases, credentials, OAuth state, caches, and histories remain in their
native locations. Agent Square reads supported history stores but never writes to them.

## SSH quick start

```bash
agent-square                 # interactive recent-thread picker
agent-square recent          # non-interactive listing
agent-square search mops
agent-square list --project mops
agent-square open <id-prefix>
agent-square scan
```

Codex, Claude Code, and OpenCode threads can be resumed in the terminal. Copilot
threads are indexed for discovery but currently need to be reopened in their native UI.

## Portability policy

- Put repository rules in the repository's `AGENTS.md`.
- Put repository-specific skills in `.agents/skills/`.
- Put reusable personal skills in `~/.agents/skills/`.
- Keep client-specific adapters thin.
- Generate MCP client configuration from a neutral declaration; do not symlink whole
  client configuration directories.
- Never store tokens in this directory. Refer to environment-variable names instead.
