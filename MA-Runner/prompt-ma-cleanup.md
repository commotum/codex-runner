You are cleaning a Markdown book export so it is easy for AI agents and humans to traverse.

Read the following inputs completely before editing:

1. The source Markdown file at `[SOURCE_MD_ABS_PATH]`
2. The book inventory JSON at `[INVENTORY_JSON_ABS_PATH]`

Then review the source carefully, paragraph by paragraph, and write a cleaned replacement to:

`[OUTPUT_MD_ABS_PATH]`

Your goals:

- Preserve the original content, structure, and meaning.
- Fix obvious OCR mistakes and formatting damage.
- Fix Markdown links so internal links point to the correct local `.md` file in the book.
- Fix external links only when the correct target is already explicit in the source or existing link text/URL.
- Make the file easier for an AI agent to read and navigate.

Allowed changes:

- Correct obvious OCR typos, dropped punctuation, duplicated characters, broken capitalization, and malformed Markdown.
- Remove obvious OCR junk such as repeated headers/footers, dangling page-number fragments, or line-break artifacts when clearly accidental.
- Normalize headings, bullet lists, block quotes, tables, and emphasis only when the current formatting is visibly broken.
- Repair internal Markdown links using the inventory. Use relative links from the source file to the real target file.
- Preserve valid external links. If an external URL is malformed but the intended URL is obvious from the surrounding text, fix it.

Do not:

- Rewrite arguments, compress sections, paraphrase for style, or modernize the author's voice.
- Add new claims, citations, sections, or summaries that are not already supported by the source.
- Invent external URLs or browse for missing sources.
- Edit any file other than `[OUTPUT_MD_ABS_PATH]`.

Output requirements:

- Write the full cleaned Markdown document to `[OUTPUT_MD_ABS_PATH]`.
- Keep the file in Markdown.
- Preserve the document order.
- If a link target cannot be determined confidently, leave the visible text in place and avoid inventing a bad link.

When deciding whether to change text, prefer restraint. Only change content when it is clearly an OCR or formatting mistake.
