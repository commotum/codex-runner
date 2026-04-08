Ok, here's the big picture. We are trying to setup up a codex runner (Runner.md) or pipleine (Pipeliner.md) using Codex-SDK.md that can convert my lecture notes from class into a custom knowledge graph, and topic lessons so that I can use active learning, direct instruction, spaced repetition, etc.

The runner/pipeline will involve several phases, and on some of them we might be trying something new. In particular some steps in our pipeline are rather context heavy. After extensive experimentation I have decided that some things just can't be prompted without an example to really illustrate things. However, full blown examples have lots of tokens and use up our compute very quickly. So I want you to look at the thread/conversation capabilities of the codex sdk to see if some of the steps could be chained together in a single conversation thread, does that make sense?

Alright here it goes, our pipeline should have the following stages:

1. Ingest - A new/fresh runner is 