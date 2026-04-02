# Preprocess Runner

`script-preprocess.py` prepares one OCR-extracted book folder for cleanup.

It follows the repo runner conventions:

- Uses `codex exec --full-auto` in non-interactive mode.
- Builds explicit prompt inputs with absolute paths.
- Writes a small manifest before work starts.
- Leaves the canonical source Markdown untouched.
- Produces deterministic on-disk outputs for the next stage.

Expected outputs in the book folder:

- `_structured/`: extracted section-level Markdown tree
- `book_progress.json`: section inventory and status
- `preprocess_manifest.json`: run manifest
- `preprocess.log`: run log

Example:

```bash
python md-Cleanup/0-Preprocess/script-preprocess.py \
  --book-dir 'md-Cleanup/3-Books-In/A New Kind of Science'
```
