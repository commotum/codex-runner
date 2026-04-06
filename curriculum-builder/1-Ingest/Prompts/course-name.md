If the input is a directory name (already correctly formatted with dash-separated words), produce a single CSV line with:

1. The course-id (a 3 character identifier for the course)
2. The course-name (the same name converted to clean English with spaces and title case)

Match the course-id and naming style to the entries in curriculum-builder/0-Source/Courses.csv where applicable; otherwise infer a reasonable 3-character code and formatted name.

A. Develop a 3-character all-caps course-id using a mnemonic code derived from the course name that is short, recognizable, and unique. Prefer initials from the main words (Continuous Time Signal Processing → CTS), and use numbers only if needed for sequence or distinction.

B. Generate the course-name by converting the directory name into a properly formatted English title: replace dashes with spaces and apply title case (Discrete-Time-Signal-Processing → Discrete Time Signal Processing).

Output only one CSV line per input.