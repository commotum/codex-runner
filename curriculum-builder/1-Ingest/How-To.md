Yes. I rebuilt it around your correction: **mastery** lives inside the lesson as step-by-step progression, and **deliberate practice** is the repeated training on the exact move being learned, not the system around it.

I also put the full version into a reusable markdown file here:

[How-to-Create-a-Perfect-Lesson-Guide.md](sandbox:/mnt/data/How-to-Create-a-Perfect-Lesson-Guide.md)

## The core rule

A perfect lesson is not a miniature chapter. It is a controlled sequence that teaches **one new move** clearly, makes the student **do it immediately**, and then strengthens it through **small, deliberate variations**.

A lesson is working when it does six things at once:

1. It teaches **one thing at a time**.
2. It says that thing **plainly**.
3. It moves from explanation to action **immediately**.
4. It builds **one small step at a time**.
5. It changes **one feature at a time**.
6. It repeats the core move enough that it starts to feel **familiar instead of fragile**.

That is what the major principles look like inside a lesson:

* **Active learning**: the student starts doing the work almost immediately.
* **Direct instruction**: the rule, pattern, or procedure is stated explicitly.
* **Deliberate practice**: the practice stays tightly focused on the exact skill being trained.
* **Mastery learning**: each micro-step becomes the foundation for the next micro-step.
* **Minimizing cognitive load**: only one new wrinkle is added at a time.
* **Developing automaticity**: the lesson gives enough correct repetitions for the move to become easier and faster.

## How to build the lesson

### 1. Reduce the topic to one teachable move

Most topics are too large to be taught well in one lesson.

Bad:

* **Complex Numbers**
* **Convolution**
* **Laplace Transforms**

Good:

* **Identify the real and imaginary parts of a complex number**
* **Determine the interval of overlap when one signal slides across another**
* **Apply the time-shifting rule to a transformed signal**

The question is: **what exact thing will the student be able to do by the end?**

That matters because a lesson that teaches multiple new moves at once becomes blurry. A lesson that teaches one move can be explicit, narrow, and receivable.

### 2. Write the lesson target and the recognition cue

The lesson target says what the student will learn to do.

Example:

> By the end of this lesson, you will be able to identify the real and imaginary parts of a complex number.

The recognition cue says when the move applies.

Example:

> Use this when a number contains a real part and a multiple of (i), or can be rewritten in that form.

That matters because students need both the move and the cue for using it.

### 3. Set the prerequisite floor

List the minimum things the student must already know.

Example:

* Know what (i) means
* Be able to simplify basic expressions
* Recognize the form (a + bi)

Keep this brief. If too many prerequisites are missing, the topic is still too large and should be split.

That matters because each step in the lesson should rest on something already secure.

### 4. Choose the cleanest possible first case

The first example should be the canonical case, not a messy one.

Good first cases:

* (3 + 4i), not (-7i + 2)
* ((3x + 1)^2), not (\sin(5x^2 - 3))
* a short clean sentence, not a paragraph full of exceptions

That matters because the first example should reveal the structure of the move, not bury it.

### 5. Build the lesson out of micro-steps

A perfect lesson is not:

* long explanation
* then a worksheet

It is:

* short explanation
* worked example
* immediate mirrored practice
* short answer check
* then one new wrinkle

Each micro-step should contain:

1. a short explanation
2. one fully worked example
3. one or two mirrored practice items
4. a brief answer check

That matters because the student should perform the move right after seeing it.

### 6. Use mirrored practice before varied practice

A mirrored question asks the student to do the **same intellectual move** as the example, with only the surface details changed.

Example:

* Worked example: identify the real and imaginary parts of (3 - 5i)
* Mirrored practice: identify the real and imaginary parts of (7 + 2i)
* Slightly varied practice: identify the real and imaginary parts of (-4i + 6)

That matters because the student first needs to feel the pattern before handling variation.

### 7. Change only one thing at a time

A good progression might be:

* Step 1: read the parts directly from (a + bi)
* Step 2: reorder the terms and still read the parts
* Step 3: handle the case where one part is zero

A bad progression would be:

* Step 1: read the parts
* Step 2: multiply complex numbers
* Step 3: graph them in the complex plane

That matters because each new step should feel like a reachable extension of the last one.

### 8. Delay edge cases until the main pattern is stable

Teach the normal case first. Then bring in exceptions.

Example:

* first teach how to read the real and imaginary parts from (a + bi)
* then handle numbers like (5) or (-2i), where one part is zero

That matters because exceptions make sense faster once the main rule is already visible.

### 9. End with a short mastery check

After the guided steps, give 3 to 5 independent problems in the **same skill family**.

The mastery check should:

* remove the immediate prompting
* keep the same core move
* avoid surprise jumps in difficulty

That matters because the lesson should end by confirming readiness, not by opening a new front.

### 10. End with a compressed summary

A good summary answers three things:

* **When do I use this?**
* **What do I do?**
* **What should I not forget?**

Example:

* Use this when a number is written in, or can be rewritten in, the form (a + bi).
* Rewrite it into (a + bi), then read off the real and imaginary parts.
* If one part is missing, that part is (0).

That matters because the student should leave holding a compact reusable rule.

## What the AI must not do

Do not let it:

* teach a whole topic family in one lesson
* explain too long before the first question
* introduce a new method inside practice
* use ugly numbers before the pattern is visible
* change notation or wording without reason
* give practice without answer checks
* end with a summary that names the topic but does not tell the student what to do

## Reusable prompt for the AI

```text
You are writing a single markdown lesson for the topic: [TOPIC].

Goal: create a lesson that a student can receive on the first pass. The lesson must introduce one new skill clearly, demonstrate it with a worked example, and then strengthen it through immediate mirrored practice and small controlled variations.

Assume that decisions about what topic comes next and when the student progresses are handled elsewhere. Your job is only to build this one lesson well.

Follow these rules:

1. Reduce the topic to one teachable move if needed.
   Why: one lesson should teach one new move, not a whole chapter.
   Example: if the topic is "Complex Numbers," narrow it to "Identify the real and imaginary parts of a complex number" if that is the cleanest first lesson.

2. State the lesson target in one sentence using "By the end of this lesson, you will be able to..."
   Why: the student needs the exact move up front.

3. Include a short "When to Use This" section.
   Why: the student must learn when the move applies, not just how.

4. Include only the minimum prerequisite reminders needed for the lesson to make sense.
   Why: every step should build on something already secure, but the lesson should not reteach the whole subject.

5. Explain the core idea briefly and explicitly.
   Why: the student should not have to infer the rule or guess the intended procedure.
   Constraint: keep the explanation short enough to hold in mind.

6. Build the lesson as 2 to 4 micro-steps.
   Each micro-step must include:
   - a short explanation,
   - one fully worked example,
   - one or two mirrored practice items,
   - a brief answer check.
   Why: this makes the lesson active, focused, and fluent.

7. Sequence the micro-steps from the simplest canonical case to slightly more advanced variants.
   Why: the lesson should progress one step at a time.
   Constraint: change only one feature at a time.

8. Keep notation, wording, and section structure consistent across the whole lesson.
   Why: consistency reduces unnecessary mental load.

9. Introduce edge cases only after the main pattern is stable.
   Why: the student needs the base case first.

10. End with a short mastery check of 3 to 5 problems in the same skill family.
    Why: the student should do the full move without prompts before the lesson ends.

11. End with a short summary that says:
    - when to use the skill,
    - what to do,
    - what to watch out for.
    Why: the lesson should compress into a reusable rule.

12. Keep the tone clear, concrete, and economical.
    Constraints:
    - no historical background,
    - no motivational filler,
    - no unrelated enrichment,
    - no discussion of broader instructional systems or implementation details,
    - no mention of other learning strategies by name inside the lesson.

Before finalizing, check:
- Is there exactly one new skill?
- Does the first practice item mirror the worked example?
- Does each later step add only one new wrinkle?
- Can the student do something immediately after each explanation?
- Is the summary procedural rather than abstract?

Return only the lesson in markdown.
```

## Reusable markdown lesson template

```md
# [Lesson Title]
<!-- Use a narrow title that names the actual skill, not the whole chapter. -->
<!-- Example: Identifying the Real and Imaginary Parts of a Complex Number -->

## Lesson Target
By the end of this lesson, you will be able to [perform one specific skill].
<!-- Why: the student needs the exact move up front. -->
<!-- Example: By the end of this lesson, you will be able to identify the real and imaginary parts of a complex number. -->

## When to Use This
[Describe the cue or pattern that tells the student this skill applies.]
<!-- Why: the student must learn when the move is relevant. -->
<!-- Example: Use this when a number contains a real part and a multiple of i, or can be rewritten in that form. -->

## What You Need First
- [Prerequisite 1]
- [Prerequisite 2]
<!-- Why: each new step should rest on something already secure. Keep this short. -->

## Core Idea
[Explain the new idea in 2–5 sentences. State the rule, relationship, or procedure explicitly. Define any new term the first time it appears.]
<!-- Why: the lesson should be explicit without becoming long or dense. -->

## Step 1: [Simplest canonical case]

### Explanation
[Explain the first micro-step in the cleanest possible case.]

### Worked Example
[Show the full solution, one step per line.]

### Try It
1. [Mirrored problem]
2. [Mirrored problem]

### Answer Check
1. [Answer]
2. [Answer]
<!-- Why: the student should perform the exact move immediately after seeing it. -->

## Step 2: [Same skill, one new wrinkle]

### Explanation
[Keep the same core move. Add only one new complication.]

### Worked Example
[Show the full solution clearly.]

### Try It
1. [Mirrored problem]
2. [Mirrored problem]

### Answer Check
1. [Answer]
2. [Answer]
<!-- Why: each new step should grow directly out of the last one. -->

## Step 3: [Same skill, slightly more advanced variant or common edge case]

### Explanation
[Introduce the next reachable variation of the same skill.]

### Worked Example
[Show the reasoning fully.]

### Try It
1. [Mirrored problem]
2. [Mirrored problem]

### Answer Check
1. [Answer]
2. [Answer]
<!-- Why: repeated work on the same move with controlled variation strengthens the skill. -->

## Watch Out
[Name one likely mistake and correct it.]
<!-- Why: if there is a predictable confusion, make it visible before it hardens. -->
<!-- Omit this section if there is no likely confusion. -->

## Mastery Check
1. [Independent problem]
2. [Independent problem]
3. [Independent problem]
4. [Optional]
5. [Optional]
<!-- Why: the student should now do the full move without immediate prompting. -->

## Answers
1. [Answer]
2. [Answer]
3. [Answer]
4. [Answer]
5. [Answer]

## Summary
- Use this when [cue].
- Do this by [procedure].
- Remember that [single key warning or pattern].
<!-- Why: end by compressing the lesson into a short reusable rule. -->
```

## Default lesson shape

If the AI is unsure how long to make the lesson, this is a good default:

* 1 sentence for the lesson target
* 1 to 2 sentences for when to use the skill
* 2 short prerequisite bullets
* 2 to 5 sentences for the core idea
* 3 micro-steps
* 1 worked example and 2 practice items per micro-step
* 3 to 5 mastery-check problems
* 3-bullet summary

That is usually enough to teach one move clearly without bloating the lesson.

The attached markdown file has the same guide in a cleaner reusable format: [How-to-Create-a-Perfect-Lesson-Guide.md](sandbox:/mnt/data/How-to-Create-a-Perfect-Lesson-Guide.md)
