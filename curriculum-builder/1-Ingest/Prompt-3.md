When extracting lesson-sized skills from lecture notes, it can help to look for material such as key facts students should know cold, important equivalences or alternate forms, mappings between representations, concepts, procedures, strategies or heuristics, representation skills, interpretation skills, common failure points or misconceptions, likely bottlenecks or missing prerequisites, and end-to-end performance capabilities. Not every lecture will contain all of these, but they are useful lenses for deciding which skills are distinct enough to extract as standalone items.


Ok, here's what we're gonna do. We need a runner built according to the best practices found in Codex-SDK.md and Runner.md and we're gonna call it ingest (so we'll have prompt-ingest.md and script-ingest.py). This will serve as the first stage of our curriculum-builder pipeline which should follow all the best practices according to Pipeliner.md. But for now let's do the following

script-ingest.py

The ingestion script has three major phases/components. It is responsible for:
1. Source management
2. Target Acquisition
3. Topic Generation (Per source document)

Source Management:
script-ingest.py starts by comparing the top level folders of curriculum-builder/0-Source against the entries in curriculum-builder/0-Source/Courses.csv 


1. start by scanning curriculum-builder/0-Source for any new documents or folders (course folders are kept within curriculum-builder/0-Source) by comparing against the course index found here: curriculum-builder/0-Source/Courses.csv
2. If there are any new course folders it needs to 
    - A. develop a 3 character all caps course-id using a 3-character mnemonic code derived from the English course name that is short, recognizable, and unique within the index. Prefer clear abbreviations from the main words (Pre Algebra → PAL), use numbers for grade or sequence levels when needed (Algebra I → AG1), and include a leading identifier for variants like honors or AP (AP Calculus AB → CAB), keeping the result compact while preserving an obvious connection to the course name.
    - B. 
    
    normalize the directory name by 