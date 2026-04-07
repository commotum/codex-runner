You are registering one top-level source folder into the course index.

Read the existing course index completely:

`[COURSES_CSV_ABS_PATH]`

The folder basename to register is:

`[COURSE_FOLDER_BASENAME]`

Return JSON only, matching the provided schema, with:

- `course_id`
- `course_name`

Requirements:

- Use only the folder basename and the existing CSV.
- If an existing CSV row already corresponds to the same course after normalizing its `course_name` into dash-separated lowercase words, reuse that exact `course_id` and `course_name`.
- Otherwise generate a new `course_id` that is exactly 3 characters, all caps letters or digits, mnemonic, recognizable, and unique within the existing CSV.
- Prefer initials from the main words when natural, and use digits only if needed for distinction.
- Generate `course_name` by converting the folder basename into a clean English title: replace dashes with spaces and use title case while preserving meaningful numerals or abbreviations.
- The resulting `course_name` must normalize back to the same folder basename.
- Do not output CSV, markdown, commentary, or extra fields.
