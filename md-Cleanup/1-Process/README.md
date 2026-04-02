# Process Runner

`script-process.py` cleans the extracted section files for one book.

It expects preprocess output in the book folder:

- `_structured/`
- `book_progress.json`

It follows the runner conventions in the repo:

- uses `codex exec --full-auto`
- operates one concrete queue item at a time
- generates an inventory of local Markdown targets
- writes to temp files and atomically promotes them
- records completion state back into `book_progress.json`

Example:

```bash
python md-Cleanup/1-Process/script-process.py \
  --book-dir 'md-Cleanup/3-Books-In/A New Kind of Science'
```
