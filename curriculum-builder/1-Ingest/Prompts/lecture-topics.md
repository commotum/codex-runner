You are extracting lesson-sized skill entries from one lecture note for the ingest runner.

Inputs
- Source lecture Markdown: `[SOURCE_MD_ABS_PATH]`
- For context only: the runner will convert your JSON response into the final per-document CSV at `[TOPICS_CSV_ABS_PATH]`

Task
Read the source lecture completely and extract a clean list of lesson-sized skills. Include prerequisite or supporting skills only when the lecture clearly teaches, invokes, or assumes them.

What each entry must represent
Each entry must be one teachable, testable learning objective that could become one full lesson. It should naturally support:
- one exact lesson target
- one recognizable `when to use this` cue
- one clean first worked example
- 2 to 4 micro-steps within the same skill family
- mirrored practice on the same move
- a mastery check that tests the same move

Conceptual row pattern
Treat this as a conceptual check, not literal output formatting:

`[short, specific skill title] – [lowercase action verb] [exact thing the learner should be able to do or understand]`

Hard constraints
- Use only the source lecture for content decisions.
- Do not invent skills that are not clearly supported by the lecture.
- Keep each entry to one clear skill or concept.
- Make each entry narrow enough to be teachable and testable.
- Prefer teachable lesson boundaries over lecture-heading boundaries.
- If the lecture repeats an idea, keep the clearest single entry.
- If a style preference conflicts with clarity or lesson-sized granularity, choose clarity and granularity.

Title rules
- Output the cleaned title text only.
- Keep it short, specific, and teachable, usually 5 to 9 words.
- Make it read like a competency label, not pasted lecture notes.
- Prefer gerund-led titles such as `Recognizing...`, `Converting...`, `Extracting...`, `Interpreting...`, `Reading...`, `Connecting...`, or `Using...`.
- A compact noun phrase is acceptable only if it still reads like a skill name.
- Do not end the title with a period.

Description rules
- Output the cleaned description text only.
- Make it exactly one sentence.
- Start it with a lowercase action verb such as `read`, `expand`, `identify`, `view`, `interpret`, `connect`, `understand`, `determine`, or `treat`.
- State the exact learning outcome or action.
- Keep it short and controlled, usually 9 to 17 words.
- Focus on one cognitive move only.
- Keep the tone neutral, instructional, concrete, and precise.
- Use notation only when it clearly anchors the concept.
- End with a period.

Prefer entries that
- already read like `By the end of this lesson, you will be able to...`
- have an obvious recognition cue or usage pattern
- have a clean canonical first case with small one-wrinkle-at-a-time variations
- allow mirrored practice immediately after a worked example
- keep the mastery check in the same skill family
- capture important facts to know cold, key equivalences, mappings between representations, procedures, interpretation moves, or prerequisite bottlenecks when clearly supported by the lecture

Exclude
- motivational filler, summaries, and administrative notes
- broad topic headings, chapter names, and topic families
- multi-step workflows that still need to be split into smaller lessons
- merged entries that combine adjacent but distinct skills
- duplicate entries for the same idea

Output format
Return JSON only. Do not output markdown, bullets, commentary, CSV, or any text outside the JSON object.

Return exactly this schema:
{
  "entries": [
    {
      "title": "string",
      "description": "string"
    }
  ]
}

Validation
- The root object must contain exactly one field: `entries`.
- Each entry must contain exactly two fields: `title` and `description`.
- `title` must not end with a period.
- `description` must be exactly one sentence and must end with a period.
- Keep formatting, cadence, and abstraction level parallel across entries.
