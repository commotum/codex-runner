Extract the lecture notes into a clean list of lesson-sized skills, including prerequisite skills. Write each item as a single learning-objective bullet in this exact format: **[short skill title]** – [one-sentence description]. The title should name one skill only, not a topic dump, and should usually read like a competency label such as “Recognizing…,” “Converting…,” “Interpreting…,” “Reading…,” “Connecting…,” or “Using…”. Keep the title short, specific, and teachable. After the dash, write one concise sentence that starts with a lowercase action verb and states exactly what the learner should be able to do or understand.

Keep every bullet narrow, concrete, and parallel in structure, tone, and level of specificity. Each description should express one cognitive move only—such as identify, interpret, convert, extract, connect, determine, or use—and should describe capability rather than merely naming subject matter. Do not stack multiple unrelated ideas into one bullet, do not write conversationally, and do not include commentary, pedagogy, or formatting variation. The result should read like a uniform set of lesson objectives: conceptually precise, compact, and easy to turn into standalone lessons.

When extracting lesson-sized skills from lecture notes, it can help to look for material such as key facts students should know cold, important equivalences or alternate forms, mappings between representations, concepts, procedures, strategies or heuristics, representation skills, interpretation skills, common failure points or misconceptions, likely bottlenecks or missing prerequisites, and end-to-end performance capabilities. Not every lecture will contain all of these, but they are useful lenses for deciding which skills are distinct enough to extract as standalone items.

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