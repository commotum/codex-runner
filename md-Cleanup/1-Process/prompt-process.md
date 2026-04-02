You are cleaning one extracted Markdown section from an OCR-derived book so it is easy for AI agents and humans to traverse.

Read these inputs completely before editing:

1. The source section Markdown at `[SOURCE_MD_ABS_PATH]`
2. The book progress JSON at `[PROGRESS_JSON_ABS_PATH]`
3. The section inventory JSON at `[INVENTORY_JSON_ABS_PATH]`

Then write a cleaned replacement to:

`[OUTPUT_MD_ABS_PATH]`

Goals:

- Preserve original meaning, order, and scope.
- Fix obvious OCR mistakes and formatting damage.
- Fix internal Markdown links so they point to the correct local section files when the target is determinable from the inventory.
- Preserve local image links that already resolve.
- Make the section clean and reliable for downstream AI traversal.

Allowed changes:

- Correct obvious OCR typos, merged words, duplicated tokens, malformed punctuation, broken capitalization, and broken Markdown syntax.
- Remove obvious OCR junk such as repeated running headers or isolated page-number fragments when clearly accidental.
- Normalize broken headings, block quotes, bullet lists, tables, and code blocks only when formatting is visibly damaged.
- Repair internal links using the provided inventory.
- Repair clearly malformed external links only when the intended target is already explicit in the source text.

Do not:

- Rewrite arguments for style.
- Add new claims, summaries, or citations.
- Invent URLs or link targets.
- Edit any file other than `[OUTPUT_MD_ABS_PATH]`.

Special handling:

- If this section is the table of contents or another navigation page, prefer clean local Markdown links to actual section files from the inventory.
- If a link target cannot be resolved confidently, keep the visible text and avoid inventing a broken link.
- Keep image paths relative and local if they already point to copied assets within the section folder.

Output requirements:

- Write the full cleaned Markdown to `[OUTPUT_MD_ABS_PATH]`.
- Keep the document in Markdown.
- Preserve section boundaries.
- Do not emit commentary instead of editing the file.
