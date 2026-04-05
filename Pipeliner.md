# Codex Runner Pipelines

Best practices for codex-runner pipelines: multi-stage systems that compose
focused runners into a reliable, resumable workflow for higher-level tasks.

## How pipelines work
- Break a large objective into a small number of ordered stages.
- Give each stage one job with one clear success condition.
- Make each stage produce deterministic on-disk outputs for the next stage.
- Let the pipeline runner coordinate stage order, retries, logging, and final
  promotion of outputs.
- Prefer a thin `main.py` entrypoint that only launches the pipeline runner.

## Core design principle
- LLM stages work best when they do one thing at a time.
- Keep each stage narrow enough that the prompt can describe the task, inputs,
  and output requirements without ambiguity.
- Give the model only the context needed for that stage, not the whole project.
- Move deterministic work out of prompts and into code whenever possible.
- Use the pipeline to turn one hard fuzzy task into several smaller precise
  tasks.

## Stage boundaries
- Split stages by function, not by convenience.
- A good stage boundary separates different kinds of work, for example:
  - discover structure
  - extract units
  - clean one unit
  - classify one unit
  - assemble final outputs
- Put semantic inference in one stage and mechanical file operations in another.
- Keep stages coarse enough to be meaningful, but fine enough to be testable and
  restartable.
- If a stage needs very different context or instructions from the next stage,
  that is usually a real pipeline boundary.

## Handoff contracts
- Treat every stage output as an interface contract.
- Prefer explicit artifacts such as:
  - manifest JSON
  - progress/state JSON
  - inventory JSON
  - extracted Markdown files
  - structured output directories
- Use stable field names and stable relative paths so later stages do not need
  to guess.
- Record authoritative status fields such as `extracted`, `cleaned`,
  `classified`, or similar stage-specific booleans or hashes.
- Use JSON Schema or another strict format when a stage must return structured
  data for code to parse.

## Orchestrator responsibilities
- The pipeline runner should discover work, launch stages, and manage control
  flow.
- Keep orchestration logic in code, not in model prompts.
- For each target item, run stages in a fixed order and stop on the first hard
  failure.
- Support single-target runs for debugging and full-queue runs for production.
- Support `--dry-run` and `--overwrite` at the pipeline level and pass them down
  to child runners when appropriate.
- Move or promote completed outputs only after all required stages succeed.

## Reliability and resumability
- Make every stage idempotent or safely restartable.
- Persist progress after each successful stage or unit of work.
- Skip work when state or content hashes prove the target is already complete.
- Write files to temp paths and atomically promote them with `os.replace`.
- Validate outputs before promotion: file exists, minimum size, parseable JSON,
  required fields present, and so on.
- Log stage starts, stage ends, commands, failures, and promotions with
  timestamps.
- Keep failure domains small so a bad item does not corrupt unrelated items.

## Context discipline
- Do not ask one prompt to both infer structure and rewrite content if those are
  separable jobs.
- Prefer an early stage that reduces ambiguity for later stages.
- Give later stages inventories, manifests, or progress files instead of the
  entire original corpus.
- Keep prompts explicit about allowed changes, forbidden changes, and exact
  output location.
- If a later stage can operate on one chapter, section, record, or file at a
  time, queue those units separately.

## Division of labor between code and Codex
- Use Codex for judgment calls, semantic extraction, cleanup, classification,
  and link repair.
- Use code for queue building, path resolution, hashing, retries, schema
  enforcement, directory creation, file copying, and atomic promotion.
- Do not rely on the model for filesystem bookkeeping that code can do
  deterministically.
- Prefer model outputs that are easy for code to validate and normalize.

## Layout and naming conventions
- Keep each runner in its own folder with its script, prompt, schema, and
  README.
- Number stage folders to make execution order obvious, for example
  `0-Preprocess`, `1-Process`, `2-Pipeline`.
- Keep raw inputs and promoted outputs in separate top-level folders.
- Use stable names for intermediate directories such as `_structured` or another
  clearly scoped working tree.
- Name artifacts by role, for example `book_progress.json`,
  `process_inventory.json`, or `preprocess_manifest.json`.

## Good pipeline patterns
- Structure first, transformation second, promotion last.
- Build a canonical inventory before asking later stages to repair links or
  references.
- Let one stage narrow the search space for the next.
- Preserve the original source tree until the pipeline has produced verified
  outputs.
- Use the filesystem as the message bus between stages when that keeps the
  system simpler and more inspectable.

## Smells to avoid
- One giant prompt that tries to discover structure, rewrite content, classify
  outputs, and organize files in a single pass.
- Stage outputs that are implicit, ad hoc, or only recoverable from logs.
- Later stages that must rediscover facts an earlier stage already knew.
- Prompts that require full-project context when a local inventory would do.
- Pipelines that mutate source data before downstream stages are validated.
- Hidden coupling where one stage depends on undocumented naming behavior from
  another.

## Decision rule
- Add a new stage when it reduces ambiguity, shrinks context, or creates a
  reusable artifact that materially improves downstream reliability.
- Do not add a stage if it only shuffles data around without simplifying the
  next job.
