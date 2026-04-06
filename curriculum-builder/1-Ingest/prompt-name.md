If the input is a directory name only, and I want the output to be:

1. The course-id (a 3 letter/number/char serving as the identifier for the course)
2. The course-name (a basic english version of the course name, nicely formatted like in the examples below)
3. The new normalized directory name (the english version but wiht dashes instead of spaces)

course-id,course-name
4GM,4th Grade Math
5GM,5th Grade Math
PAL,Pre Algebra
AG1,Algebra I
GEO,Geometry
AG2,Algebra II
IM1,Integrated Math I
IM2,Integrated Math II
IM3,Integrated Math III
PCL,Pre Calculus
HI1,Honors Integrated Math I
HI2,Honors Integrated Math II
HI3,Honors Integrated Math III
CAB,AP Calculus AB
CBC,AP Calculus BC
SMF,SAT Math Fundamentals
MF1,Mathematical Foundations I
MF2,Mathematical Foundations II
MF3,Mathematical Foundations III
CA1,Calculus I
CA2,Calculus II
MVC,Multivariable Calculus
DEQ,Differential Equations
PAS,Probability and Statistics
LAL,Linear Algebra
DSM,Discrete Mathematics
MOP,Methods of Proof
MML,Mathematics for Machine Learning

Convert a directory name into CSV format with three fields: course-id (3-character code), course-name (clean English title), and normalized-name (lowercase, dash-separated version of course-name). Match course-id and naming style to this list where applicable; otherwise infer a reasonable 3-character code and formatted name. Output only one CSV line.


A. Develop a 3-character all-caps course-id using a mnemonic code derived from the English course name that is short, recognizable, and unique within the index. Prefer clear abbreviations from the main words (Pre Algebra → PAL), use numbers for grade or sequence levels when needed (Algebra I → AG1), and include a leading identifier for variants like honors or AP (AP Calculus AB → CAB), keeping the result compact while preserving an obvious connection to the course name.

B. Generate a clean, properly formatted English course name from the directory name using standard spacing and capitalization. Expand abbreviations and remove noise while preserving meaning (pre-alg → Pre Algebra), and ensure the result reads naturally as a course title.

C. Normalize the directory name into a consistent, dash-separated format based on the cleaned course name, preserving capitalization. Replace spaces with dashes and remove irregular characters (Probability and Statistics → Probability-and-Statistics), ensuring consistency even if the original directory name is messy.