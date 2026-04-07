You are extracting lesson-sized topic rows from one lecture note for the ingest runner.

Read the source lecture Markdown completely:

`[SOURCE_MD_ABS_PATH]`

The runner will write the final per-document CSV to:

`[TOPICS_CSV_ABS_PATH]`

Task:

Extract the lecture notes into a clean list of lesson-sized skills, including prerequisite skills. At the row level, the target pattern is:

`[short, specific skill title] – [lowercase action verb] [exact thing the learner should be able to do or understand]`

The title should name one skill only, not a topic dump, and should usually read like a competency label such as `Recognizing...`, `Converting...`, `Interpreting...`, `Reading...`, `Connecting...`, or `Using...`. Keep the title short, specific, and teachable. After the dash, write one concise sentence that starts with a lowercase action verb and states exactly what the learner should be able to do or understand.

Keep every row narrow, concrete, and parallel in structure, tone, and level of specificity. Each description should express one cognitive move only, such as identify, interpret, convert, extract, connect, determine, or use, and should describe capability rather than merely naming subject matter. Do not stack multiple unrelated ideas into one row, do not write conversationally, and do not include commentary, pedagogy, or formatting variation. The result should read like a uniform set of lesson objectives: conceptually precise, compact, and easy to turn into standalone lessons.

Style requirements:

- Write learning objectives, not raw topic labels.
- Describe capability, not just subject matter.
- Keep each row narrow enough to be teachable and testable.
- Keep rows similar in size, rhythm, and abstraction level.
- Avoid stacking multiple unrelated ideas into one row.
- Treat each row as one two-part unit: title, then dash, then one-sentence description.
- The title must name one skill or concept only.
- The title should usually be about 5 to 9 words.
- The title should read like a competency label, not like notes pasted from the lecture.
- Prefer gerund-led titles such as `Recognizing...`, `Converting...`, `Extracting...`, `Interpreting...`, `Reading...`, `Connecting...`, or `Using...`.
- A compact noun-phrase title is acceptable only when it still reads like a skill name, for example `Real-axis projection of a phasor`.
- The description should start immediately after the dash with a lowercase action verb such as `read`, `expand`, `identify`, `view`, `interpret`, `connect`, `understand`, `determine`, or `treat`.
- The description must explain the exact learning outcome or action, not just name the topic area.
- Keep descriptions short and controlled, usually about 9 to 17 words.
- Keep each description focused on one cognitive move only.
- Keep the tone neutral, instructional, and concrete.
- Keep the language conceptually precise without becoming bloated.
- Use notation only when it anchors the concept clearly.
- End every description with a period.
- Keep formatting, cadence, and grammatical structure parallel across rows.

Downstream requirement:

These rows are not just labels. Each row should already be suitable to become one full lesson under the lesson template used in the next stage. Only include rows that can naturally support:

- one exact lesson target
- one recognizable "when to use this" cue
- one clean first worked example
- 2 to 4 micro-steps that stay in the same skill family
- mirrored practice on the same move
- a mastery check that still tests the same move

In other words, extract lesson seeds, not broad topics.

Selection rules:

- Use only the source lecture and the style examples above.
- Include skills directly taught by the lecture.
- Include prerequisite or supporting skills only when the lecture clearly relies on them, invokes them explicitly, or assumes them for understanding.
- Keep exactly one clear skill or concept per row.
- Prefer rows that already read like `By the end of this lesson, you will be able to...`.
- Prefer rows with an obvious recognition cue or usage pattern.
- Prefer rows with a clean canonical first case and small one-wrinkle-at-a-time variations.
- Prefer rows that would allow mirrored practice immediately after a worked example.
- Prefer rows whose mastery check would stay in the same skill family rather than opening a new front.
- Do not include motivational filler, repeated summaries, administrative notes, or broad topic headings that are not themselves teachable skills.
- Do not include broad umbrellas such as chapter names, topic families, or multi-step workflows that would need to be split again before lesson writing.
- Do not merge two adjacent skills into one broader row just because they appear together in the lecture.
- If the lecture repeats an idea in multiple places, keep the clearest single row for that idea.
- Prefer teachable row boundaries over lecture-heading boundaries.
- Consider facts to know cold, important equivalences, mappings between representations, procedures, interpretation moves, and likely prerequisite bottlenecks when they are genuinely supported by the lecture.

Output contract:

- Return JSON only, matching the provided schema.
- The root object must contain exactly one field: `entries`.
- Each entry must contain exactly two fields:
  - `title`
  - `description`
- Do not output markdown bullets, CSV, commentary, or any text outside the JSON object.

Field rules:

- `title` is the cleaned bullet title only, without `**`.
- `description` is the cleaned sentence after the dash only, without the dash.
- `title` must not end with a period.
- `description` must be exactly one sentence and must end with a period.
- Preserve useful notation such as `$t=0$` only when it is important to the skill.
