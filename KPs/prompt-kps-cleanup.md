You are normalizing a small batch of knowledge-point bullets so they match an existing target style exactly.

Read the source Markdown file completely:

`[SOURCE_MD_ABS_PATH]`

Your job is to convert every bullet in that file into one structured CSV-style row with:

- `title`
- `description`

The source file contains exactly [EXPECTED_ENTRY_COUNT] list items. Your output must contain exactly [EXPECTED_ENTRY_COUNT] entries in the same order.

Use this target style reference. These rows are the authoritative examples to imitate:

```csv
[TARGET_STYLE_REFERENCE_CSV]
```

Target structure:

- Each source bullet becomes exactly one output row.
- Preserve the original order.
- Do not drop, merge, split, or invent items.
- Preserve the original meaning and scope, but rewrite the phrasing so it matches the target style.

Title rules:

- Output only the cleaned title text.
- No bullet markers.
- No numbering.
- No markdown bold markers.
- No trailing period.
- Keep it short and specific, usually about 4 to 10 words.
- Prefer a competency-style label, often gerund-led, when that is natural.
- A compact noun phrase is acceptable when it matches the target style better.
- Inline notation such as `$t=0$` is allowed only when it is important to the concept.

Description rules:

- Output only the cleaned description text.
- Write exactly one sentence.
- Start with a lowercase action verb when possible.
- End with a period.
- Keep it compact and concrete, usually about 8 to 20 words.
- State the actual learning objective or interpretation, not just the topic area.
- Preserve useful notation when it clarifies the concept.

Stylistic rules:

- Match the tone of the target examples: neutral, instructional, precise, compact.
- Keep one clear skill or concept per row.
- Avoid note-like phrasing, fragments, meta commentary, and redundant filler.
- Do not include markdown formatting, code fences, bullets, numbering, or explanation outside the schema.

Return JSON only, matching the provided schema.
