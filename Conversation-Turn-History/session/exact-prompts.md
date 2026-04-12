# Exact Prompts

Note
- These are the exact user prompts recovered from `~/.codex/history.jsonl` up through the prompt that created `Conversation Turn History.md`.
- Recoverable patches are attached to the most recent prompt before each patch timestamp.

## Prompt 1

**Timestamp**: `1775598395`

**Text**

Can you read the full contents of new/0-Context/How-To.md new/0-Context/Perfect-Lesson-Template.md and new/0-Context/Perfect-Lesson.md and then tell me what you've learned?

**Recoverable Patches**

- None

## Prompt 2

**Timestamp**: `1775599604`

**Text**

Ok, now check out all three lessons in new/0-Context/Lessons

**Recoverable Patches**

- None

## Prompt 3

**Timestamp**: `1775599662`

**Text**

No, these are examples of a perfect lesson.

**Recoverable Patches**

- None

## Prompt 4

**Timestamp**: `1775599729`

**Text**

why are active learning and direct instruction so important?

**Recoverable Patches**

- None

## Prompt 5

**Timestamp**: `1775599767`

**Text**

and how do they play a role in our lesson format?

**Recoverable Patches**

- None

## Prompt 6

**Timestamp**: `1775599811`

**Text**

and how are the exemplar lessons different than the template in its current state?

**Recoverable Patches**

- None

## Prompt 7

**Timestamp**: `1775600057`

**Text**

Ok, can you create two .md files in new/2-Lesson-Gen one called Lesson-Template.md and another called Lesson-Prompt.md that more accurately reflect the exemplars in terms of what we're trying to create?

**Recoverable Patches**

- `patches/by-prompt/prompt-07/01.patch` -> /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Template.md, /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Prompt.md

## Prompt 8

**Timestamp**: `1775600237`

**Text**

Can you tell me, in new/0-Context/Lessons/14.md what is the difference between each step in the lesson? like the table of contents is segmented into steps, but what makes each step different?

**Recoverable Patches**

- None

## Prompt 9

**Timestamp**: `1775600342`

**Text**

is this more in line with our earlier lesson template? or the new one?

**Recoverable Patches**

- None

## Prompt 10

**Timestamp**: `1775600416`

**Text**

Can you update new/2-Lesson-Gen/Lesson-Template.md and new/2-Lesson-Gen/Lesson-Prompt.md to reflect that?

**Recoverable Patches**

- `patches/by-prompt/prompt-10/01.patch` -> /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Template.md, /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Prompt.md

## Prompt 11

**Timestamp**: `1775600699`

**Text**

Based on our new template, how many topics or lessons should be built out from the contents of new/0-Context/Week 1/1.1 Periodic Signals.md ?

**Recoverable Patches**

- None

## Prompt 12

**Timestamp**: `1775600959`

**Text**

Can you put those topics as written into new/0-Context/Week 1/1.1 Topics.md

**Recoverable Patches**

- `patches/by-prompt/prompt-12/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 13

**Timestamp**: `1775601005`

**Text**

whoah, I said as written. Where's all the other stuff you wrote? like where's the core move, etc

**Recoverable Patches**

- `patches/by-prompt/prompt-13/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 14

**Timestamp**: `1775601157`

**Text**

Can you make it more like 
`[short, specific skill title] – [lowercase action verb] [exact thing the learner should be able to do or understand]`
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

**Recoverable Patches**

- `patches/by-prompt/prompt-14/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 15

**Timestamp**: `1775601229`

**Text**

oh no, you deleted all the other stuff. I still want that other stuff, but with the updated titles

**Recoverable Patches**

- `patches/by-prompt/prompt-15/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 16

**Timestamp**: `1775601255`

**Text**

what about the core move?

**Recoverable Patches**

- `patches/by-prompt/prompt-16/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 17

**Timestamp**: `1775601430`

**Text**

Ok, now can you use the template to create an outline of the sections for each of the lessons? I don't want the full lesson, just an outline of what will change for each section, and what needs to be covered. Does that make sense? Remember what changed between each section in lesson 14

**Recoverable Patches**

- `patches/by-prompt/prompt-17/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Lesson Outlines.md

## Prompt 18

**Timestamp**: `1775601572`

**Text**

we shouldn't restrict or force three lesson blocks. Unless it's exactly what we need.

**Recoverable Patches**

- `patches/by-prompt/prompt-18/01.patch` -> /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Template.md, /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Prompt.md, /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Lesson Outlines.md

## Prompt 19

**Timestamp**: `1775602021`

**Text**

fewest is too strict as well. It should be exactly the full progression. We shouldn't try to keep the section count down, nor try to meet a quota. This is the distribution for Math Academy. • Using the bullet items under each lesson’s ## Table of Contents, I parsed 2608
  lessons and counted 13929 total TOC sections. The average is 5.34 sections per
  lesson.

  Section-count distribution:

  - 3 sections: 182 lessons
  - 4 sections: 606 lessons
  - 5 sections: 743 lessons
  - 6 sections: 590 lessons
  - 7 sections: 298 lessons
  - 8 sections: 117 lessons
  - 9 sections: 46 lessons
  - 10 sections: 21 lessons
  - 11 sections: 3 lessons
  - 12 sections: 1 lesson
  - 13 sections: 1 lesson
so the vast majority of our lessons should be 4-7 sections long with a minimum of 3 sections and let's say a maximum of 8.

**Recoverable Patches**

- `patches/by-prompt/prompt-19/01.patch` -> /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Template.md, /home/jake/Developer/codex-runner/new/2-Lesson-Gen/Lesson-Prompt.md, /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Lesson Outlines.md

## Prompt 20

**Timestamp**: `1775602816`

**Text**

Can you also generate a list of prerequisite math topics needed for each of the selected topics?

**Recoverable Patches**

- `patches/by-prompt/prompt-20/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 21

**Timestamp**: `1775603210`

**Text**

Without looking in the files, which of the following courses likely has relevant prerequisites to our topics? new/0-Context/MA-Course-Maps

**Recoverable Patches**

- None

## Prompt 22

**Timestamp**: `1775603344`

**Text**

Ok, read the full maps from the most likely courses and find pre-existing prerequisites.

**Recoverable Patches**

- None

## Prompt 23

**Timestamp**: `1775603485`

**Text**

yes please

**Recoverable Patches**

- `patches/by-prompt/prompt-23/01.patch` -> /home/jake/Developer/codex-runner/new/0-Context/Week 1/1.1 Topics.md

## Prompt 24

**Timestamp**: `1775930412`

**Text**

can you see the conversation history of this session?

**Recoverable Patches**

- None

## Prompt 25

**Timestamp**: `1775930467`

**Text**

Can you create a turn by turn summary of the conversation we had and place it in "Conversation Turn History.md" ?

**Recoverable Patches**

- `patches/by-prompt/prompt-25/01.patch` -> /home/jake/Developer/codex-runner/Conversation Turn History.md
