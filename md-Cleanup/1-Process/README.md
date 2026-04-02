# MA Runner

`script-ma-cleanup.py` is a resumable Codex runner for cleaning the Markdown files in `The Math Academy Way`.

What it does:

- Walks the book tree recursively and processes every `.md` file.
- Generates a fresh inventory of local Markdown targets before each run.
- Asks Codex to read each file carefully, paragraph by paragraph.
- Limits edits to OCR cleanup, broken Markdown repair, and link correction.
- Writes to a temp file first, then atomically replaces the source file.
- Records completed file hashes in a state file so reruns can skip unchanged files.

Files:

- `script-ma-cleanup.py`: the runner
- `prompt-ma-cleanup.md`: the cleanup instructions sent to Codex
- `math-academy-inventory.json`: generated at runtime
- `.ma_cleanup_state.json`: generated state
- `ma_cleanup.log`: generated run log

Usage:

```bash
python MA-Runner/script-ma-cleanup.py
```

Useful flags:

```bash
python MA-Runner/script-ma-cleanup.py --dry-run
python MA-Runner/script-ma-cleanup.py --overwrite
python MA-Runner/script-ma-cleanup.py --source 'FRONT-MATTER/Contents/Contents.md'
python MA-Runner/script-ma-cleanup.py --source 'III-COGNITIVE-LEARNING-STRATEGIES'
```

Notes:

- The runner defaults to in-place cleanup of the source Markdown files.
- Internal links are intended to resolve to real local `.md` files under `The Math Academy Way`.
- External links are only corrected when the target is already determinable from the file itself.
