You are preparing a single OCR-extracted book folder for downstream Markdown cleanup.

Read all of these inputs completely before making changes:

1. The source Markdown file at `[SOURCE_MD_ABS_PATH]`
2. The destination progress JSON path at `[PROGRESS_JSON_ABS_PATH]`
3. The destination structured root directory at `[STRUCTURED_ROOT_ABS_PATH]`

Context:

- The source Markdown is a large OCR extraction for one book.
- The book folder may also contain the original PDF, metadata JSON files, and extracted images.
- The source Markdown is the canonical extraction source and must remain unchanged.

Your job:

- Inspect the source Markdown carefully and identify the contents / table of contents and the leaf sections that should become separate Markdown files.
- Create a structured output tree rooted at `[STRUCTURED_ROOT_ABS_PATH]`.
- Create and populate `[PROGRESS_JSON_ABS_PATH]`.
- Extract one Markdown file per leaf section into the structured tree.
- Copy only the images actually referenced by each extracted section into that section's local `Images/` folder and rewrite local image links accordingly.
- Replace the extracted `Contents.md` with a clean navigation file that links to all extracted section files in the structured tree.

Required behavior:

- Do not modify `[SOURCE_MD_ABS_PATH]`.
- Do not move, rename, or delete original images, PDFs, or metadata files.
- Create or update only files under `[STRUCTURED_ROOT_ABS_PATH]` and `[PROGRESS_JSON_ABS_PATH]`.
- If the structured tree already exists, inspect it and update it carefully rather than duplicating or clobbering valid work unnecessarily.

Extraction rules:

- Leaf extraction units are front matter items, numbered chapters, FAQ sections, appendices, notes, glossary, references, index, and similar terminal sections.
- Major parts should become grouping folders when they are clearly present.
- Folder names should be deterministic, readable slugs derived from the heading text.
- Preserve leading chapter numbers where present.
- Convert `&` to `and`.
- Remove Markdown formatting markers and simplify punctuation.
- Markdown filenames must match their containing folder names, except `Contents.md`, `Preface.md`, `Index.md`, and similar named sections may keep the natural section name if it also matches the folder.

Progress JSON requirements:

- Create `[PROGRESS_JSON_ABS_PATH]` as valid JSON.
- It must contain:
  - `book_title`
  - `source_markdown`
  - `structured_root`
  - `generated_at`
  - `sections`
- `sections` must be an array of objects.
- Each section object must contain:
  - `id`
  - `title`
  - `group`
  - `folder`
  - `markdown_path`
  - `source_heading`
  - `extracted`
  - `cleaned`
- Set `extracted` to `true` for sections successfully created.
- Set `cleaned` to `null` for every section at preprocess time.

Image handling:

- If an extracted section references local images from the source book folder, create `Images/` under that section folder.
- Copy only the images used by that section.
- Rewrite image paths in that section Markdown to point to `Images/<filename>`.
- Do not invent image references.

Contents handling:

- Extract the source contents section if possible, but the final `Contents.md` should be a clean navigation page built from the extracted section inventory.
- Use relative Markdown links from the generated `Contents.md` to the actual local section files.

Validation:

- Confirm there is one Markdown file per extracted section.
- Confirm referenced copied images exist.
- Confirm links in the rewritten `Contents.md` resolve.
- Confirm the progress JSON matches the created output tree.

Output:

- Perform the file creation and updates directly on disk.
- Do not print a giant report.
- If some ambiguous sections cannot be confidently split, prefer conservative extraction and keep section boundaries aligned to explicit headings.
