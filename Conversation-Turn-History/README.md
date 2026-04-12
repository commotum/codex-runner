# Conversation-Turn-History Bundle

This folder is a standalone analysis bundle for the session summarized in `Conversation Turn History.md`.

It includes:
- the turn summary itself,
- the exact user prompt stream up through the summary-creation step,
- current copies of every referenced file that could be identified,
- directory listings for referenced directories,
- recoverable `apply_patch` diffs extracted from the Codex log for the same session,
- Git history dumps for the referenced files.

Key limits:
- The exact user prompts are recoverable from `~/.codex/history.jsonl`.
- The exact assistant prose is not fully recoverable for every turn; the most reliable artifacts are the summary, the prompts, and the patch hunks.
- Some file references are marked as inferred when the summary referred to a directory rather than naming each file explicitly.
- The summary turns and the exact prompt stream are separated because they do not line up perfectly one-to-one late in the session.

Counts:
- Summary turns: 26
- Exact prompts captured before the bundle cutoff: 25
- Referenced files copied: 16
- Referenced directories listed: 2
- Recoverable `apply_patch` hunks before the bundle cutoff: 13

Suggested reading order for another AI:
1. `source/Conversation Turn History.md`
2. `session/turns-01-26.md`
3. `session/exact-prompts.md`
4. `patches/patch-index.json`
5. `files/current/`
6. `git-history/`
