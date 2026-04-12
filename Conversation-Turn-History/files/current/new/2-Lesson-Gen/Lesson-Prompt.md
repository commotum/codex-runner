You are writing a single markdown lesson for the topic: [TOPIC].

Goal: create a production-style lesson page that matches the exemplar lessons. The lesson should stay inside one coherent topic family and teach it through a sequence of tightly related sections. The outer structure should look like a full topic lesson page, but the internal progression should still behave like a controlled step-by-step lesson: clean base case first, then the same move with one new wrinkle at a time.

Assume that decisions about what lesson comes before or after this one are handled elsewhere. Your job is to create one polished lesson page in the established house style.

Follow these rules:

1. Keep the lesson inside one coherent topic family.
   Usually, the lesson should revolve around one underlying move, method, or very tight skill family, even if it includes several sections.
   Why: the exemplars are unified topic pages, not mixed-topic worksheets.

2. Build the lesson as a sequence of named sections sized to the content.
   The section count should reflect the full progression of the lesson.
   In this system, most lessons should land between 4 and 7 TOC sections, with a normal minimum of 3 and maximum of 8.
   Use natural section titles that name the content or task, such as:
   - "Using Euler's Formula for Planar Graphs"
   - "Computing a Conditional Probability"
   - "Applying Subdivisions to a Graph"
   Do not use generic headings like "Step 1" or "Step 2."
   Even so, those sections should function like steps in a progression.
   Why: the exemplars organize by topic-specific subskills and ideas, but the order still matters instructionally.

3. Include a Table of Contents and a Prerequisites section near the top.
   Include an Introduction only if the student needs a short setup, central definition, or key formula before the first main section.
   Why: this matches the production lesson layout.

4. Let the lesson cover multiple closely related subskills when the topic needs it.
   Each section should still have a clear local focus, and adjacent sections should feel like reachable extensions of one another.
   Why: the exemplars are broader than a one-move micro-lesson, but each block remains controlled.

5. Sequence the lesson from the cleanest foundational case to controlled variants, then to edge cases, applications, or proofs.
   Change only one meaningful feature at a time when possible.
   If a later section is more advanced, it should clearly rest on what the student just learned.
   Use enough sections to show the full progression clearly.
   Do not compress away a real intermediate step, and do not add a fake step just to increase section count.
   Why: the page-level format is broad, but the learning progression should still be incremental.

6. Most skill or application sections should include:
   - one worked example,
   - a direct explanation,
   - one or two immediate questions.
   Short concept, theorem, or definition sections may omit questions if their job is to prepare the next application section.
   Why: the exemplars mix exposition sections with example-and-practice sections.

7. In each example-and-practice section, the first question should mirror the worked example as closely as possible.
   A second question may add a small surface variation, but should not introduce a different move.
   Why: the exemplars teach through controlled repetition, not abrupt jumps.

8. Keep notation, phrasing, and formatting consistent across the whole lesson.
   Number questions consecutively across the entire lesson.
   Why: consistency reduces unnecessary friction.

9. Use the question format that best fits the topic.
   Acceptable formats include:
   - multiple choice,
   - fill in the blank,
   - table-based choices,
   - image-based prompts,
   - short guided proof starters.
   If the topic depends on a diagram, graph, or visual, include a clear image placeholder line in markdown.
   Why: the exemplars use production-ready question formats rather than one fixed drill format.

10. Keep the explanation clear, direct, and economical.
   Constraints:
   - no historical background,
   - no motivational filler,
   - no discussion of pedagogy,
   - no meta commentary about lesson design,
   - no unrelated enrichment.
   Why: the lesson should stay focused on content and action.

11. Do not force sections that belong to the old micro-lesson template.
    By default, avoid sections such as:
    - "Lesson Target"
    - "When to Use This"
    - "Core Idea"
    - "Watch Out"
    - "Mastery Check"
    - "Answers"
    - "Summary"
    Use them only if the user explicitly asks for them.
    Why: the exemplars are section-driven content pages, not pedagogy-labeled handouts.

12. Use anchor tags and anchored Table of Contents links when listing sections.
    Why: the exemplars use navigable section structure.

Before finalizing, check:
- Does the lesson read like one coherent topic page?
- Does each section have a clear local focus?
- Does the section count reflect the full progression rather than a compressed or padded version?
- Does the lesson stay in the normal 3 to 8 section range, with 4 to 7 as the usual target, unless the content strongly justifies otherwise?
- Does the lesson move from foundation to controlled variation or higher application?
- Does each new skill block preserve the same underlying move or extend it in one clear way?
- Are worked examples placed before nearby practice where needed?
- Does the first practice item in each block closely mirror the example?
- Are the headings, question types, and formatting natural for the topic?

Return only the lesson in markdown.
