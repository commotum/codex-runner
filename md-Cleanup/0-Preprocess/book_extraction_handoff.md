Use this as the handoff prompt in a fresh context:

You are working in a local OCR-extraction workspace for a single book that was converted from PDF into Markdown plus extracted image files.

The input folder structure will typically look like this:

- A book folder, for example `fin/Some Book Title/`
- Inside that folder:
  - one main source Markdown file, usually named like the book title, for example `Some Book Title.md`
  - the original PDF, for example `Some Book Title.pdf`
  - a metadata file, often `Some Book Title_meta.json`
  - many extracted image files referenced by the Markdown, usually filenames like `_page_123_Figure_4.jpeg` or `_page_87_Picture_2.jpeg`

Your job is to turn that single large OCR Markdown file into a clean, navigable folder tree of section-level Markdown files, one per front-matter section / chapter / FAQ / back-matter section, while preserving images and rewriting image links so every extracted Markdown file is self-contained relative to its own folder.

Follow this process exactly:

1. Locate the source book folder and identify the main source Markdown file.
- Do not modify the source Markdown file.
- Treat that source Markdown file as the canonical source for all extraction work.

2. Inspect the source Markdown structure before creating anything.
- Read the heading structure with line numbers.
- Identify:
  - book title
  - contents section
  - top-level parts/sections such as `I. PRELIMINARIES`, `III. COGNITIVE LEARNING STRATEGIES`, etc.
  - leaf sections that should become their own folders and files:
    - front matter items like `Contents`, `Preface`
    - numbered chapters like `Chapter 4. Core Technology: the Knowledge Graph`
    - FAQ sections like `FAQ: Student Behavior`
    - back matter items like `Glossary`, `References`, `Notes for Future Additions`
- The actual extraction units are the leaf sections, not every subheading inside a chapter.

3. Create a new structured subdirectory inside the book folder.
- The new subdirectory usually has the same name as the book, nested one level deeper, for example:
  - source: `fin/The Math Academy Way/`
  - structured output root: `fin/The Math Academy Way/The Math Academy Way/`
- Inside this new structured root, create high-level grouping folders that mirror the source organization, for example:
  - `FRONT-MATTER`
  - `I-PRELIMINARIES`
  - `II-ADDRESSING-CRITICAL-MISCONCEPTIONS`
  - `III-COGNITIVE-LEARNING-STRATEGIES`
  - `IV-COACHING`
  - `V-TECHNICAL-DEEP-DIVES`
  - `VI-FREQUENTLY-ASKED-QUESTIONS`
  - `BACK-MATTER`

4. Build the leaf folder structure from the source headings.
- Under each grouping folder, create one folder per leaf extraction unit.
- Folder names should be normalized from the actual heading text.
- Use consistent slug formatting:
  - preserve leading chapter numbers for chapters, e.g. `4-Core-Technology-the-Knowledge-Graph`
  - use `FAQ-...` for FAQ sections, e.g. `FAQ-Student-Behavior`
  - convert `&` to `and`
  - remove formatting markers like `**`
  - remove or simplify punctuation
  - replace spaces with hyphens
  - keep names readable and deterministic
- Example:
  - heading `Chapter 14. Minimizing Cognitive Load`
  - folder `14-Minimizing-Cognitive-Load`

5. Extract the contents section first.
- Find the exact line range for the source `Contents` section.
- Extract that range into:
  - `.../FRONT-MATTER/Contents/Contents.md`
- At this stage, it may still be OCR-messy. That is fine temporarily.
- The source file remains unchanged.

6. Extract every other leaf section from the source Markdown.
- For each leaf section:
  - determine its start line at the relevant heading
  - determine its end line as the line before the next leaf section begins
- Write the extracted text into a Markdown file inside that section’s folder.
- The Markdown filename must match the containing folder name exactly.
- Example:
  - folder: `.../III-COGNITIVE-LEARNING-STRATEGIES/16-Layering/`
  - file: `16-Layering.md`

7. Handle images for each extracted section.
- Scan each extracted section for local Markdown image references, e.g.:
  - `![](_page_69_Figure_2.jpeg)`
  - or other relative local image paths
- For each extracted section that references local images:
  - create an `Images` subfolder inside that section folder
  - copy only the images referenced by that section into that `Images` folder
  - update image links in the extracted Markdown so they point to the new local path:
    - from `![](_page_69_Figure_2.jpeg)`
    - to `![](Images/_page_69_Figure_2.jpeg)`
- Do not move the original images.
- Do not modify the source Markdown.
- If a section has no local images, do not invent an `Images` folder unless needed.

8. Keep every extracted section self-contained.
- Each extracted `.md` file should work relative to its own directory.
- That means:
  - local image references must point to its own `Images` subfolder
  - filenames should match folder names
  - the file should contain the full extracted text for that leaf section only

9. Clean up and rewrite the extracted `Contents.md`.
- Do not try to preserve a broken OCR table of contents if it is badly fragmented.
- Instead, replace the extracted `Contents.md` with a clean Markdown outline that links to all extracted section files in the new structured directory.
- This new clean contents page should include:
  - Preface
  - all chapter links grouped under their major part headings
  - all FAQ links
  - Glossary / References / Notes for Future Additions
- Use relative links from `FRONT-MATTER/Contents/Contents.md` to the actual extracted files.
- Example relative link:
  - `../../III-COGNITIVE-LEARNING-STRATEGIES/14-Minimizing-Cognitive-Load/14-Minimizing-Cognitive-Load.md`

10. Validate everything after extraction.
- Confirm there is one `.md` file per leaf section folder.
- Confirm each `.md` filename matches its folder name.
- Confirm all relative links in `Contents.md` resolve.
- Confirm all local image references inside extracted sections resolve.
- Spot-check a few representative sections:
  - one with many images
  - one with one image
  - one with no images
  - one FAQ
  - one back-matter section

11. Important constraints.
- Never modify the original source Markdown file.
- Never rename or delete original source image files.
- Never overwrite unrelated user files.
- Only create or update files inside the new structured output tree unless explicitly asked otherwise.
- If a destination folder already exists, inspect it before writing. Avoid duplicating work or clobbering valid existing content unless instructed.
- If OCR formatting is messy inside extracted text, preserve the source text unless the user explicitly asks for editorial cleanup. The exception is `Contents.md`, which should be rebuilt cleanly as a navigation file.
- Keep chapter boundaries based on actual heading structure, not guessed page numbers.

12. Naming rules to follow consistently.
- Group folder names should mirror the book’s major sections.
- Leaf folder names should be stable, human-readable slugs derived from heading text.
- Leaf Markdown filenames should exactly match leaf folder names.
- FAQ files should match FAQ folder names.
- Back matter files should match their folder names.
- `Contents.md` stays named `Contents.md`.
- `Preface.md` stays named `Preface.md`.

13. Typical end state.
Given a source folder like:
- `fin/Book Title/Book Title.md`
- source images in `fin/Book Title/`

You should end with:
- `fin/Book Title/Book Title/FRONT-MATTER/Contents/Contents.md`
- `fin/Book Title/Book Title/FRONT-MATTER/Preface/Preface.md`
- `fin/Book Title/Book Title/I-PRELIMINARIES/1-.../1-....md`
- etc.
- and for sections with images:
  - `.../<Section Folder>/Images/<copied image files>`

14. Preferred workflow in practice.
- First inspect headings and line numbers.
- Then create the destination tree.
- Then extract one or two sample sections to confirm the pattern.
- Then automate the full extraction across all leaf folders.
- Then rewrite `Contents.md`.
- Then validate links and images.

15. Deliverable summary.
When finished, report:
- the structured output root you created
- how many leaf section folders were populated
- whether `Contents.md` was rewritten as a clean linked outline
- whether image links were rewritten and copied locally for image-bearing sections
- whether the original source Markdown remained untouched

If you want, I can also turn this into a shorter “operator prompt” version optimized to paste directly into a fresh Codex session.
