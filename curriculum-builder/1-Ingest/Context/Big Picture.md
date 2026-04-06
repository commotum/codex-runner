Ok, here's what we're gonna do. We need a runner built according to the best practices found in Codex-SDK.md and Runner.md and we're gonna call it ingest. It has multiple prompts for various portions of the runner that are all context dependent and will be activated by various triggers detected by our single runner script script-ingest.py. The prompts will be found in curriculum-builder/1-Ingest/Prompts . This will serve as the first stage of our curriculum-builder pipeline which should follow all the best practices according to Pipeliner.md. But for now let's do the following

script-ingest.py

The ingestion script has three major phases/components. It is responsible for:
1. Source management
2. Target Acquisition
3. Topic Generation (Per source document)

Source Management:

1. script-ingest.py starts by comparing the top level folders of curriculum-builder/0-Source against the entries in curriculum-builder/0-Source/Courses.csv and if there are any new folders to be added then it launches a single runner per new folder with the curriculum-builder/1-Ingest/prompt-name.md prompt. This prompt needs to be revised and polished so that it fits the standards of Runner.md and our other best practices. You'll need to read it and fix it to understand how it handles the course index Courses.csv.
2. After all top level folders are registered in the index, script-ingest.py should then compare the full contents of each course recursively all the way through and comparing the folder's contents with the document index found in the root of that folder. 








1. start by scanning curriculum-builder/0-Source for any new documents or folders (course folders are kept within curriculum-builder/0-Source) by comparing against the course index found here: curriculum-builder/0-Source/Courses.csv
2. If there are any new course folders it needs to 
    - A. develop a 3 character all caps course-id using a 3-character mnemonic code derived from the English course name that is short, recognizable, and unique within the index. Prefer clear abbreviations from the main words (Pre Algebra → PAL), use numbers for grade or sequence levels when needed (Algebra I → AG1), and include a leading identifier for variants like honors or AP (AP Calculus AB → CAB), keeping the result compact while preserving an obvious connection to the course name.
    - B. 
    
    normalize the directory name by 