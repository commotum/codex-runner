PART 1 - WE'VE COMPLETED THIS ALREADY, THIS IS JUST A COPY OF WHAT I ASKED FOR.

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

---

PART 2 - THIS IS WHAT WE NEED TO FINISH RIGHT NOW. THIS IS STILL PART OF OUR SINGLE RUNNER, 1 SCRIPT, SEVERAL PROMPTS.

Now that the administrative stuff has been handled, it's time to dig into our star player, the topic generation runners.

For each document in the set list script-ingest.py we launch a new runner in a new codex session and feed it the prompt from curriculum-builder/1-Ingest/Prompts/lecture-topics.md Right now that prompt is very rough, messy, and all over the place. it needs to be brought into compliance with Runner.md and Codex-SDK.md best practices. It's a hodge-podge of a few different prompts we built including for a runner we built earlier.

The runner we built earlier simply took messy raw LLM .md outputs for lesson topics and made them uniform and added them to a csv. However, the adding to the csv part and uniformity is important even if our source isn't a simple .md list of topics, if that makes sense.

Importantly, you need to craft both the prompt and the script such that our runner if given the "curriculum-builder/0-Source/Continuous-Time-Signal-Processing/Lectures/Week 1/1.2 Complex Exponentials.md" file would have produced the following topics for our topics .csv:

"Recognizing a complex exponential signal","read $z(t)=Ae^{j(\omega_0 t+\phi)}$ as a signal with magnitude, angular frequency, and phase."
"Converting a complex exponential to rectangular form","expand the signal into explicit real and imaginary pieces."
"Extracting the real and imaginary waveforms","identify the cosine waveform as the real part and the sine waveform as the imaginary part."
"Interpreting a complex exponential in the complex plane","view the signal as a complex quantity with magnitude and angle that change meaningfully over time."
"Reading phasors at $t=0$","interpret $z(0)=Ae^{j\phi}$ as the initial phasor with starting angle $\phi$."
"Interpreting angular frequency as rotation rate","connect $\omega_0$ to how fast the phasor turns in the complex plane."
"Real-axis projection of a phasor","understand the cosine component as the projection of the rotating phasor onto the real axis."
"Imaginary-axis projection of a phasor","understand the sine component as the projection of the rotating phasor onto the imaginary axis."
"Reading projection values at special angles","determine when the real or imaginary component becomes zero or reaches an extreme from the total angle."
"Connecting phasor motion to time-domain sinusoids","understand why a rotating phasor traces sine and cosine waves over time."
"Using complex exponentials as compact sinusoid representations","treat exponential form as a cleaner representation for analysis and later signal-processing work."
"Reading sinusoid parameters","identify amplitude, angular frequency, and phase in expressions like $A\cos(\omega t+\phi)$ or $A\sin(\omega t+\phi)$."
"Interpreting radians and standard angles","interpret angles like $\pi/2$ and $\pi$ as positions in a rotation."
"Identifying real and imaginary parts","distinguish the real component of a complex number from its imaginary component."
"Interpreting a complex number in the plane","view a complex number as a point or vector on the real and imaginary axes."
"Describing magnitude and angle","describe a complex number by its length and direction."
"Applying Euler's formula to exponentials","rewrite exponential form as cosine plus $j$ sine, and move between the two forms."

The script should generate a single topics .csv per document, per runner. It should place that .csv file in curriculum-builder/0-Source/Continuous-Time-Signal-Processing/Topics and use the title/name convention of "[DOCUMENT-ID]-Topics.csv" and each csv should have the columns "tile" and "description" Sound good?

Can you help me fix the prompt, and add that functionality to the script? 

---

Can you help me finish part 2?