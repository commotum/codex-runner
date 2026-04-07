You are extracting lesson-sized topic rows from one lecture note for the ingest runner.

Read the source lecture Markdown completely:

`[SOURCE_MD_ABS_PATH]`

The runner will write the final per-document CSV to:

`[TOPICS_CSV_ABS_PATH]`

These CSV rows are authoritative examples of the target style, granularity, ordering, and phrasing:

```csv
[TARGET_STYLE_REFERENCE_CSV]
```

Task:

Extract a clean list of lesson-sized skills from the lecture note itself. Include both:

- skills directly taught by the lecture
- prerequisite or supporting skills that the lecture clearly relies on, invokes explicitly, or assumes for understanding

The source is a lecture note, not a prewritten topic list. Infer the rows from the lecture content.

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
- Keep exactly one clear skill or concept per row.
- Do not include motivational filler, repeated summaries, administrative notes, or broad topic headings that are not themselves teachable skills.
- If the lecture repeats an idea in multiple places, keep the clearest single row for that idea.
- Prefer the most teachable ordering, not necessarily the lecture heading order.
- Usually place central lecture skills first, then tightly related prerequisite or supporting skills, then broader enabling equivalence tools if they are needed.
- When a supported skill already appears in the style reference with a strong wording match, prefer that established wording over inventing a near-synonym.
- If a style-reference row is fully supported by the source lecture, reuse its title and description exactly rather than paraphrasing it.
- If the style reference clearly corresponds to the same lecture or the same topic family, treat its row boundaries, ordering, and wording as canonical whenever the source supports them.
- Do not collapse two supported rows into one broader row or replace a supported row with a looser paraphrase when the style reference already separates them cleanly.

Output contract:

- Return JSON only, matching the provided schema.
- The root object must contain exactly one field: `entries`.
- Each entry must contain exactly two fields:
  - `title`
  - `description`
- Do not write files, emit CSV, or include commentary outside the JSON object.

Title rules:

- Output only the cleaned title text.
- No bullet markers, numbering, markdown bold markers, or trailing period.
- Keep it short, specific, and teachable, usually about 4 to 10 words.
- Name one skill only, not a topic dump.
- Prefer a competency-style label such as `Recognizing...`, `Converting...`, `Interpreting...`, `Reading...`, `Connecting...`, `Using...`, `Extracting...`, `Identifying...`, or `Determining...` when natural.
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
- Do not include pedagogy, explanations, markdown formatting, or any output outside the schema.
