You are extracting lesson-sized topic rows from one lecture note for a curriculum ingestion runner.

Read the source lecture Markdown completely:

`[SOURCE_MD_ABS_PATH]`

These rows are authoritative examples of the target style. Match their tone, granularity, and phrasing:

```csv
[TARGET_STYLE_REFERENCE_CSV]
```

The runner will write your extracted rows into:

`[TOPICS_CSV_ABS_PATH]`

Task:

Extract a clean list of lesson-sized skills from the lecture note, including:

- skills directly taught by the note
- prerequisite or supporting skills that the note clearly relies on or invokes explicitly

The source is a lecture note, not a prewritten topic list. Infer the rows from the lecture content itself.

Useful lenses:

- key facts or formulas students should know cold
- important equivalences or alternate forms
- mappings between representations
- concepts, procedures, strategies, or heuristics
- representation skills and interpretation skills
- likely prerequisite bottlenecks or common confusions
- end-to-end performance capabilities

Selection rules:

- Use only the source lecture and the style reference above.
- Include only distinct, teachable skills that are supported by the lecture.
- Keep one clear skill or concept per row.
- Do not include motivational filler, repeated summaries, administrative notes, or broad topic headings that are not themselves teachable skills.
- If the lecture repeats an idea in multiple places, keep the clearest single row for that idea.
- Prefer the most teachable ordering, not necessarily the lecture heading order.
- Usually place central lecture skills first, then tightly related prerequisite or supporting skills, then broader enabling equivalence tools if they are needed.

Output requirements:

- Return JSON only, matching the provided schema.
- The root object must contain an `entries` array.
- Each entry must contain exactly:
  - `title`
  - `description`

Title rules:

- Output only the cleaned title text.
- No bullet markers, numbering, markdown bold markers, or trailing period.
- Keep it short, specific, and teachable, usually about 4 to 10 words.
- Name one skill only, not a topic dump.
- Prefer a competency-style label such as “Recognizing...”, “Converting...”, “Interpreting...”, “Reading...”, “Connecting...”, “Using...”, “Extracting...”, “Identifying...”, or “Determining...” when natural.
- A compact noun phrase is acceptable only when it matches the target style better.
- Preserve useful inline notation such as `$t=0$` only when it is important to the concept.

Description rules:

- Output only the cleaned description text.
- Write exactly one sentence.
- Start with a lowercase action verb when possible.
- End with a period.
- Keep it compact, concrete, and parallel to the style reference, usually about 8 to 20 words.
- State exactly what the learner should be able to do or understand.
- Express one cognitive move only, such as identify, interpret, convert, extract, connect, determine, or use.
- Preserve useful notation when it clarifies the skill.

Style rules:

- Match the target examples: neutral, instructional, precise, compact.
- Keep all rows parallel in structure, tone, and specificity.
- Do not stack multiple unrelated ideas into one row.
- Do not include commentary, pedagogy, explanations, or formatting outside the schema.
