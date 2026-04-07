Ok, here's what we're gonna do. We need a runner built according to the best practices found in Codex-SDK.md and Runner.md and we're gonna call it ingest. It has multiple prompts for various portions of the runner that are all context dependent and will be activated by various triggers detected by our single runner script script-ingest.py. The prompts will be found in curriculum-builder/1-Ingest/Prompts . This will serve as the first stage of our curriculum-builder pipeline which should follow all the best practices according to Pipeliner.md. But for now let's do the following

script-ingest.py

The ingestion script has three major phases/components. It is responsible for:
1. Source management
2. Target Acquisition
3. Topic Generation (Per source document)

Source Management:

1. script-ingest.py starts by comparing the top level folders of curriculum-builder/0-Source against the entries in curriculum-builder/0-Source/Courses.csv and if there are any new folders to be added then it launches a single runner per new folder with the curriculum-builder/1-Ingest/Prompts/course-name.md prompt. This prompt needs to be revised and polished so that it fits the standards of Runner.md and our other best practices. You'll need to read it and fix it to understand how it handles the course index Courses.csv.
2. After all top level folders are registered in the index, script-ingest.py should then compare the full contents of each course recursively all the way through and comparing the course folder's contents with the document index found in the root of that folder. If there is a new document then it should be added to the content/document index. The content index should be named like so: [COURSE-ID]-Contents.csv usng the course-id generated in step 1 and found in the courses index. If the folder doesn't have a contents index then the script should make one. The contents index should have the following columns: index, document-id, document-path, document-ingested, where the index is just a monotonically increasing 4 digit integer with leading zeroes like 0001, 0002, 0003, ... , 0102 , ... , 9999. Document id is [COURSE-ID]-SRC-[INDEX], document path is the relative path on disk to the file, and document ingested is a date or timestamp for when it was ingested, or blank if it has not been ingested yet. When the document is first added to the index it will be blank for document-ingested

Target Acquisition:

1. After the source management functions have been completed run the acquisition function
2. For all courses, and for all content indices at the root of each course folder, make a set of paths of documents that have not been ingested yet (blank under document-ingested column, with no date) to pass on to topic generation

Topic Generation:

For each document in the set list, a new runner will start a new session to generate lesson topics from it. We'll get to that next. But for now, let's finish off the first two parts. Sound good?


Now that the administrative stuff has been handled, it's time to dig into our star player, the topic generation runners.

1. For each document in the set list script-ingest.py we launch a new runner in a new codex session and feed it the prompt from curriculum-builder/1-Ingest/Prompts/lecture-topics.md Right now that prompt is very rough, messy, and all over the place. it needs to be brought into compliance with Runner.md and Codex-SDK.md best practices. It's a hodge-podge of a few different prompts we built including for a runner we built earlier. 








1. start by scanning curriculum-builder/0-Source for any new documents or folders (course folders are kept within curriculum-builder/0-Source) by comparing against the course index found here: curriculum-builder/0-Source/Courses.csv
2. If there are any new course folders it needs to 
    - A. develop a 3 character all caps course-id using a 3-character mnemonic code derived from the English course name that is short, recognizable, and unique within the index. Prefer clear abbreviations from the main words (Pre Algebra → PAL), use numbers for grade or sequence levels when needed (Algebra I → AG1), and include a leading identifier for variants like honors or AP (AP Calculus AB → CAB), keeping the result compact while preserving an obvious connection to the course name.
    - B. 
    
    normalize the directory name by 