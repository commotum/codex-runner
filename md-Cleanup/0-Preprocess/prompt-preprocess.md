You are analyzing a single OCR-extracted book Markdown file.

Read the source Markdown at:

`[SOURCE_MD_ABS_PATH]`

Your task is only to identify the book's navigational structure from its table of contents and surrounding heading structure.

Return structured JSON matching the provided schema. Do not write files. Do not output prose.

Requirements:

- Identify the book title.
- Identify the contents heading text if present.
- Identify the leaf sections that should become extracted Markdown files.
- Each section should represent a terminal extraction unit such as:
  - front matter item
  - chapter
  - appendix
  - notes
  - glossary
  - bibliography / references
  - index
  - FAQ section
- Do not include subsection titles inside a chapter as separate extraction units.
- Preserve the source order.
- Use stable readable folder slugs that match the style of `The Math Academy Way`.
- Use major grouping folders when clearly present from the contents or headings.

Slug rules:

- For numbered chapters, use `N-Title-Case-Words` style, for example:
  - `6-The-Persistence-of-Neuromyths`
  - not `chapter-6-the-persistence-of-neuromyths`
- Do not include a `chapter` prefix in the slug.
- Preserve leading chapter numbers where present.
- Use title-cased slug components rather than lowercase slugs.
- Keep small connector words lowercase where natural, such as `and`, `of`, `the`, `in`, `on`, `to`, `for`.
- Convert `&` to `and`.
- Remove Markdown formatting markers and simplify punctuation.
- Replace spaces with hyphens.
- Keep names deterministic and readable.

Group rules:

- Use a normalized group folder name such as `FRONT-MATTER`, `PART-I`, `CHAPTERS`, `APPENDICES`, `BACK-MATTER`, `FAQ`, or another clearly justified equivalent based on the source.
- If the book has explicit named parts, prefer those as groups.

Conservative behavior:

- If a section is ambiguous, prefer the clearest extraction unit rather than inventing a finer split.
- If some front matter items are listed in the contents but are not likely extractable as distinct source regions, you may still include them if they are clearly intended top-level units.

Only output JSON conforming to the schema.
