You are extracting lesson-sized skill entries from one lecture note for the ingest runner.

Inputs
- Source lecture Markdown: `[SOURCE_MD_ABS_PATH]`
- Reference lecture example: `[REFERENCE_LECTURE_MD_ABS_PATH]`
- Reference extracted topics example: `[REFERENCE_TOPICS_CSV_ABS_PATH]`
- For context only: the runner will convert your JSON response into the final per-document CSV at `[TOPICS_CSV_ABS_PATH]`

Task
Read the source lecture completely and extract a coverage-complete lesson map from it.

This is not a lecture summary task.
This is not a conceptual compression task.
This is not a heading-cleanup task.

Your job is to recover the full set of distinct lesson-sized skills and concepts that this lecture would need if its contents were turned into standalone lessons.

Coverage rule
Include:
- every distinct lesson-sized skill, concept, interpretation, procedure, equivalence, or representation change that the lecture clearly teaches
- every direct prerequisite or supporting skill that the lecture relies on to make those later skills understandable or executable

Prefer completeness over elegance.
A slightly longer but instructionally complete list is better than a short, polished list that compresses away prerequisite bottlenecks or merges multiple teachable moves into one entry.
Missing a direct prerequisite or collapsing two real lessons into one is worse than including two nearby but distinct entries.

Reference example
Read the reference lecture and its extracted topics example to understand the target level of completeness, lesson boundaries, title style, description style, and prerequisite handling.

Use the reference pair only as an example of what a good extraction looks like.
Use only the source lecture for deciding which entries belong in the current output.
Do not invent skills that are not clearly supported by the lecture.
However, do not require a prerequisite to be announced as a formal section heading before you include it. If the lecture clearly relies on a skill to read notation, interpret a diagram, understand a derivation, or execute a later procedure, and that skill is directly supported by the lecture, include it.

What each entry must represent
Each entry must be one teachable, testable learning objective that could become one full lesson.
It should naturally support:
- one exact lesson target
- one recognizable `when to use this` cue
- one clean first worked example
- 2 to 4 micro-steps within the same skill family
- mirrored practice on the same move
- a mastery check that tests the same move

A good entry is:
- one coherent cognitive move
- one interpretation
- one representation-reading skill
- one representation-conversion skill
- one rule application
- one threshold concept that later skills depend on
- or one meaningful edge case that changes the reasoning enough to deserve its own lesson

A bad entry is:
- a broad topic family
- a chapter label
- a merged bundle of adjacent but distinct skills
- a raw example copied from the notes
- a tiny surface variation that does not change the reasoning
- or a motivational takeaway instead of a teachable skill

Conceptual row pattern
Treat this as a conceptual check, not literal output formatting:

`[short, specific skill title] – [lowercase action verb] [exact thing the learner should be able to do or understand]`

How to decide whether something deserves its own entry
A candidate item should usually become its own entry if most of the following are true:
- it is one coherent move or understanding
- it can be taught in one sitting
- it can be practiced directly
- it can be checked independently
- it is distinct from neighboring items
- later skills depend on it

How to handle prerequisites and supporting skills
Include direct prerequisite or supporting skills when the lecture clearly teaches, invokes, or depends on them.

This includes skills such as:
- reading notation or symbolic form
- recognizing the structure of an expression
- identifying components, parameters, axes, or parts
- interpreting a representation geometrically or conceptually
- reading special values, key cases, or key equivalences
- converting between closely related forms
- understanding a threshold concept that later procedures rely on

Do not backchain indefinitely into remote background knowledge.
Do not include generic prior knowledge unless the lecture actually depends on it in a direct and visible way.
The right target is the immediate instructional floor beneath the lecture’s main ideas.

Important distinction:
If a later skill depends on a smaller support skill, and the lecture actively uses that support skill as part of the explanation, derivation, or interpretation, include both.
Do not include only the “main” skill and assume the prerequisite floor will be inferred later.

How to split oversized candidates
Split a candidate into separate entries when any of these are true:
- it contains more than one real verb or learning action
- it crosses more than one representation in a nontrivial way
- it combines reading with interpreting
- it combines interpreting with converting
- it combines converting with extracting
- it combines a prerequisite skill with a dependent skill
- it combines a base case with a meaningful edge case
- it would require multiple different worked examples because it is really more than one lesson

Usually keep these as separate entries when they are distinct:
- recognizing a form
- reading parameters
- identifying parts or components
- interpreting meaning
- converting representations
- applying a rule
- connecting two representations
- handling a meaningful special case

Important:
The fact that the lecture derives one idea from another does not mean they should be merged into one entry.
Lecture flow is not the same thing as lesson boundary.

How to avoid oversplitting
Do not split trivial surface variations into separate entries.
Different numbers, reordered terms, positive versus negative signs, or repeated examples usually do not justify separate entries unless the reasoning materially changes.
Prefer one entry for one underlying skill, not one entry per example.

How to treat examples
Infer the reusable skill beneath the examples.
Do not output example-specific entries.
If several examples all demonstrate the same move, keep one normalized entry for that move.

How to handle repetition
If the lecture repeats an idea in several places, keep the clearest single entry.
Merge true duplicates and near-duplicates.
But do not merge two entries merely because they are adjacent or closely related if they still represent different teachable moves.

What to prefer
Prefer entries that:
- already read like `By the end of this lesson, you will be able to...`
- expose a real prerequisite bottleneck
- have an obvious recognition cue or usage pattern
- have a clean canonical first case
- allow small one-wrinkle-at-a-time variations
- allow mirrored practice immediately after a worked example
- keep the mastery check in the same skill family
- capture important mappings, equivalences, interpretations, procedures, or component-reading moves that later ideas depend on

Do not optimize for a short elegant list.
Do not collapse the lecture into its conceptual spine.
Do not produce a compact summary of the main derivation path if that would hide lesson-sized subskills.

Exclude
- motivational filler, summaries, and administrative notes
- broad topic headings, chapter names, and topic families
- multi-step workflows that still need to be split into smaller lessons
- merged entries that combine adjacent but distinct skills
- duplicate entries for the same idea
- historical remarks or application remarks that do not introduce a distinct skill
- “why this matters” statements that are not themselves teachable objectives

Ordering
Order the final entries from foundational to dependent when possible.
A good default order is:
- prerequisite notation or recognition skills
- prerequisite interpretation skills
- core concept lessons
- direct procedures and conversions
- more dependent interpretation or connection lessons
- meaningful special cases or derived uses

Title rules
- Output the cleaned title text only.
- Keep it short, specific, and teachable, usually 4 to 9 words.
- Make it read like a lesson-sized skill name, not pasted lecture notes.
- Prefer gerund-led titles such as `Recognizing...`, `Identifying...`, `Reading...`, `Interpreting...`, `Converting...`, `Extracting...`, `Applying...`, `Connecting...`, or `Using...`.
- A compact noun phrase or title-case lesson title is acceptable only if it still clearly reads like one lesson-sized skill.
- The title must name one skill or concept only.
- Do not end the title with a period.

Description rules
- Output the cleaned description text only.
- Make it exactly one sentence.
- Start it with a lowercase action verb.
- Prefer concrete verbs such as `identify`, `read`, `rewrite`, `determine`, `evaluate`, `interpret`, `connect`, `convert`, `apply`, `recognize`, `distinguish`, `locate`, or `extract`.
- Avoid vague verbs like `understand`, `know`, `learn`, or `treat` unless no sharper verb fits.
- State the exact learning outcome or action.
- Focus on one cognitive move only.
- Keep it short and controlled, usually 8 to 18 words.
- Keep the tone neutral, instructional, concrete, and precise.
- Use notation only when it clearly anchors the concept.
- End with a period.

Selection procedure
Follow this process internally before producing the final JSON:
1. Read the lecture completely.
2. List all explicit candidate skills, concepts, procedures, interpretations, mappings, and representation changes.
3. Add the direct prerequisite or support skills that the lecture clearly relies on.
4. Split oversized candidates into smaller lesson-sized entries.
5. Merge only true duplicates or cosmetic rephrasings.
6. Remove examples, commentary, and non-lesson material.
7. Order the surviving entries from foundational to dependent.
8. Rewrite every entry into the normalized title and description style.

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
- Do not force a short list. The correct number of entries is whatever the lecture clearly supports.

Before finalizing, silently check:
- Did I produce a lesson map rather than a conceptual summary?
- Did I include the direct prerequisite floor beneath the main lecture ideas?
- Did I split entries that combine recognition, interpretation, conversion, extraction, or connection moves?
- Is each entry small enough for one lesson but large enough to justify one lesson?
- Would later entries become harder to teach if an earlier support skill were removed?
- Did I accidentally merge items just because they appear in one derivation sequence?
- Did I keep only one normalized entry per real skill?
