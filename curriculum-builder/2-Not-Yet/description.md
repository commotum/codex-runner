These three files all follow the same core architecture, and they are much closer to **curriculum maps/course outlines** than to full administrative syllabi. Each one has exactly four major sections—**Course Description**, **Course Overview**, **Course Outcomes**, and **Course Content**—and then drills downward using a consistent markdown hierarchy: `#` for the course title, `##` for major sections, `###` for outcome domains and content units, bolded module headings, and bulleted lesson/topic lines with hierarchical numbering.   

The **Description** explains where the course sits in a sequence, who it is for, what prior knowledge it builds on, and what comes next. The **Overview** is a short summary of the course’s major themes. The **Outcomes** section is competency-based: it groups learning goals into broad domains such as arithmetic, trigonometry, limits, calculus, probability, or linear algebra, and each domain contains action-oriented bullet points. The **Content** section is the most granular part: it enumerates broad units, then modules, then individual lessons/topics.   

Across the sequence, the subject matter escalates in a clean progression. **Mathematical Foundations I** covers arithmetic, fractions, ratios, the number system, exponents, equations, functions, polynomials, and foundational geometry. **Mathematical Foundations II** moves into quadratics, advanced functions, exponentials/logarithms, rational/radical functions, trigonometry, limits, introductory calculus, vectors, statistics, and probability. **Mathematical Foundations III** shifts into deeper calculus, sequences and series, conics, parametric and polar curves, advanced trigonometry, complex numbers, differential equations, 3D vectors, matrices/linear transformations, and probability distributions.   

A few structural patterns are especially worth preserving in a reusable template. First, the **outcomes are broader than the content units**; they describe what learners can do, not just what topics appear. Second, the **content numbering is hierarchical and explicit**: unit `1`, module `1.1`, topic `1.1.1`. Third, each unit heading includes a **topic count** such as “32 topics” or “50 topics,” and those counts align with the listed lessons under that unit. When generating a new outline, treat that as a final formatting step: finish the unit, count the listed topics, and then insert the topic count in the unit heading. That makes the outline readable both as a syllabus summary and as a curriculum inventory.   

Here is a reusable markdown template that matches that pattern for any course:

```md
# [Course Title]

## Course Description

[Course Title] is [the first course in a sequence / an intermediate course / an advanced course] designed for [target audience]. Building on [prior knowledge, prerequisite course, or assumed background], students in [Course Title] will develop [core knowledge areas and skills]. Upon completing this course, students will be prepared for [next course, next level, professional application, or follow-on study].

## Course Overview

[Write a short, high-level summary of the course in 1–3 sentences. Highlight the major themes, skill areas, and why the course matters.]

## Course Outcomes

Upon successful completion of this course, students will have mastered the following:

### [Outcome Domain 1]

- [Action-oriented learning outcome]
- [Action-oriented learning outcome]
- [Action-oriented learning outcome]

### [Outcome Domain 2]

- [Action-oriented learning outcome]
- [Action-oriented learning outcome]
- [Action-oriented learning outcome]

### [Outcome Domain 3]

- [Action-oriented learning outcome]
- [Action-oriented learning outcome]
- [Action-oriented learning outcome]

### [Outcome Domain 4]

- [Action-oriented learning outcome]
- [Action-oriented learning outcome]
- [Action-oriented learning outcome]

### [Outcome Domain 5]

- [Action-oriented learning outcome]
- [Action-oriented learning outcome]
- [Action-oriented learning outcome]

## Course Content

### 1. [Unit Name] [X topics]

**1.1. [module Name]**

- 1.1.1. [Topic Title]
- 1.1.2. [Topic Title]
- 1.1.3. [Topic Title]

**1.2. [module Name]**

- 1.2.1. [Topic Title]
- 1.2.2. [Topic Title]
- 1.2.3. [Topic Title]

**1.3. [module Name]**

- 1.3.1. [Topic Title]
- 1.3.2. [Topic Title]
- 1.3.3. [Topic Title]

### 2. [Unit Name] [X topics]

**2.1. [module Name]**

- 2.1.1. [Topic Title]
- 2.1.2. [Topic Title]
- 2.1.3. [Topic Title]

**2.2. [module Name]**

- 2.2.1. [Topic Title]
- 2.2.2. [Topic Title]
- 2.2.3. [Topic Title]

### 3. [Unit Name] [X topics]

**3.1. [module Name]**

- 3.1.1. [Topic Title]
- 3.1.2. [Topic Title]
- 3.1.3. [Topic Title]

**3.2. [module Name]**

- 3.2.1. [Topic Title]
- 3.2.2. [Topic Title]
- 3.2.3. [Topic Title]

### 4. [Unit Name] [X topics]

**4.1. [module Name]**

- 4.1.1. [Topic Title]
- 4.1.2. [Topic Title]
- 4.1.3. [Topic Title]

**4.2. [module Name]**

- 4.2.1. [Topic Title]
- 4.2.2. [Topic Title]
- 4.2.3. [Topic Title]

### 5. [Unit Name] [X topics]

**5.1. [module Name]**

- 5.1.1. [Topic Title]
- 5.1.2. [Topic Title]
- 5.1.3. [Topic Title]

**5.2. [module Name]**

- 5.2.1. [Topic Title]
- 5.2.2. [Topic Title]
- 5.2.3. [Topic Title]

<!-- Continue adding units as needed -->
```

To keep the result closest to the Mathematical Foundations files, use these rules:

* keep **Description** sequence-oriented and audience-oriented,
* keep **Overview** short,
* make **Outcomes** broad and skill-based,
* make **Content** detailed and hierarchical,
* when finished with each unit, count the listed topics and insert the unit topic count,

