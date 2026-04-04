Ok, sweet. Now I need your help building an orchestration pipeline for md-Cleanup with the following components:

3 runners (md-Cleanup/0-Preprocess, md-Cleanup/1-Process, md-Cleanup/2-Pipeline)
2 content folders (md-Cleanup/3-Books-In, md-Cleanup/4-Books-Out)
1 main.py script (md-Cleanup/main.py)

All runners should follow the default setup and standards as outlined in README.md and according to Codex-SDK.md

md-Cleanup/3-Books-In
- This folder contains sub-folders of OCR extractions obtained from a prior .pdf to .md program/neural network. They contain images, the original PDF, some metadata in .json files, and the extracted .md file.
- These have not been run through our cleanup process just yet

md-Cleanup/main.py
- This script only exists to initialize/launch/run the md-Cleanup/2-Pipeline runner
- It's there so it's easy for me to run the command without having to remember the directory structure or contents

md-Cleanup/2-Pipeline
- scans the md-Cleanup/3-Books-In folder to generate a list of directory targets
- then for each book directory target it:
    - launches md-Cleanup/0-Preprocess to generate the inputs for md-Cleanup/1-Process
    - launches md-Cleanup/1-Process
    - upon completion moves the target directory to md-Cleanup/4-Books-Out

md-Cleanup/0-Preprocess
- a very rough and unpolished prompt exists in this folder as md-Cleanup/0-Preprocess/book_extraction_handoff.md (You will need to rename it to match the folder name, and rewrite it so that it satisfies the guidelines as laid out in Codex-SDK.md and README.md)
- it should:
    - create an empty metadata/progress .json file for the book's folder
    - get the Contents or TOC from the books .md file (as described in the prompt)
    - populate the progress .json from the extracted contents
    - create the sub-directories named for various chapters/sections
    - extract the chapter/contents from the original .md to a new one specific to that sub folder
    - mark the chapter as extracted, leaving the "cleaned" field empty

md-Cleanup/1-Process
- a polished runner already exists in this folder, but it's tailored specifically to the math academy way book we already did. We need to make this process more general, rename the files, and rewrite the prompt and script to fit into our guidelines as laid out in Codex-SDK.md and README.md and then it should process each chapter or sub-directory with contents with a separate runner to do the cleanup job

md-Cleanup/4-Books-Out
- This folder contains sub folders of our cleaned up and polished outputs

Any questions? Any suggestions for improvement?



A useful way to think about these is as names for portions of a body of knowledge at different scales. Each level stands in for a denser web of ideas underneath it.

**1. Course**
A course is a bounded body of study within a larger discipline, treated as one coherent whole. It pulls together many related ideas, methods, and problem types under a single identity. In that sense, a course is a compressed label for a substantial region of a subject.

**2. Unit**
A unit is a major conceptual region within a course. It groups together material that belongs to the same broad theme, mode of reasoning, or family of problems. A unit is large enough to have its own internal progression, but still clearly part of the larger whole.

**3. Subunit**
A subunit is a focused branch within a unit. It gathers ideas that are tightly connected by method, dependency, or purpose. Where a unit names a major territory, a subunit names one distinct thread inside that territory.

**4. Topic**
A topic is an individual idea, method, tool, or pattern of reasoning that can be studied directly. It is specific enough to explain, practice, and assess on its own. This is usually the level where the material becomes concrete.

Another compact way to say it:

* A **course** says, “treat this whole region of the subject as one thing.”
* A **unit** says, “these major threads belong together.”
* A **subunit** says, “this is one focused thread within that larger thread.”
* A **topic** says, “this is one particular thing to learn.”

The main point is that these categories are contextual, not absolute. The same material can appear as a topic in one setting, a unit in another, and a full course somewhere else. The label tells you how much surrounding material is being bundled together at that moment.
