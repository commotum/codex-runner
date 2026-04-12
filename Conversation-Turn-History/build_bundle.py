from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/jake/Developer/codex-runner")
BUNDLE_ROOT = REPO_ROOT / "Conversation-Turn-History"
SUMMARY_PATH = REPO_ROOT / "Conversation Turn History.md"
EXTRA_OUTPUT_11 = REPO_ROOT / "Output-11.md"
CODEX_HISTORY = Path("/home/jake/.codex/history.jsonl")
CODEX_LOG = Path("/home/jake/.codex/log/codex-tui.log")

SESSION_ID = "019d69e8-bc59-7462-b7a2-80752ce57345"
TURN_LIMIT = 26
SCOPE_END_PROMPT_SNIPPET = 'place it in "Conversation Turn History.md"'


@dataclass(frozen=True)
class FileRef:
    path: str
    turns: tuple[int, ...]
    reference_type: str
    note: str


@dataclass(frozen=True)
class DirRef:
    path: str
    turns: tuple[int, ...]
    note: str


FILE_REFS = [
    FileRef(
        "new/0-Context/How-To.md",
        (1,),
        "explicit",
        "Read in full at the start of the session.",
    ),
    FileRef(
        "new/0-Context/Perfect-Lesson-Template.md",
        (1,),
        "explicit",
        "Read in full at the start of the session.",
    ),
    FileRef(
        "new/0-Context/Perfect-Lesson.md",
        (1,),
        "explicit",
        "Read in full at the start of the session.",
    ),
    FileRef(
        "new/0-Context/Lessons/14.md",
        (2, 8),
        "explicit",
        "One of the three exemplar lesson files in the directory and later analyzed directly for its section-to-section progression.",
    ),
    FileRef(
        "new/0-Context/Lessons/2754.md",
        (2,),
        "inferred",
        "Turn 2 referenced 'all three lessons' in new/0-Context/Lessons; this directory currently contains only 14.md, 2754.md, and 3003.md.",
    ),
    FileRef(
        "new/0-Context/Lessons/3003.md",
        (2,),
        "inferred",
        "Turn 2 referenced 'all three lessons' in new/0-Context/Lessons; this directory currently contains only 14.md, 2754.md, and 3003.md.",
    ),
    FileRef(
        "new/2-Lesson-Gen/Lesson-Template.md",
        (7, 10, 19, 20),
        "explicit",
        "Created and revised multiple times during the session.",
    ),
    FileRef(
        "new/2-Lesson-Gen/Lesson-Prompt.md",
        (7, 10, 19, 20),
        "explicit",
        "Created and revised multiple times during the session.",
    ),
    FileRef(
        "new/0-Context/Week 1/1.1 Periodic Signals.md",
        (11,),
        "explicit",
        "Source lecture used to derive lesson topics.",
    ),
    FileRef(
        "new/0-Context/Week 1/1.1 Topics.md",
        (12, 13, 14, 15, 16, 17, 21, 24),
        "explicit",
        "Primary file that changed repeatedly while refining topic phrasing and prerequisites.",
    ),
    FileRef(
        "new/0-Context/Week 1/1.1 Lesson Outlines.md",
        (18, 19, 20),
        "explicit",
        "Created and refined as the outline target.",
    ),
    FileRef(
        "new/0-Context/MA-Course-Maps/Mathematical-Foundations-I.md",
        (23,),
        "explicit",
        "Read to identify exact pre-existing prerequisites.",
    ),
    FileRef(
        "new/0-Context/MA-Course-Maps/Mathematical-Foundations-II.md",
        (23,),
        "explicit",
        "Read to identify exact pre-existing prerequisites.",
    ),
    FileRef(
        "new/0-Context/MA-Course-Maps/Mathematical-Foundations-III.md",
        (23,),
        "explicit",
        "Read to identify exact pre-existing prerequisites.",
    ),
    FileRef(
        "new/0-Context/MA-Course-Maps/Differential-Equations.md",
        (23,),
        "explicit",
        "Read to identify exact pre-existing prerequisites.",
    ),
    FileRef(
        "Conversation Turn History.md",
        (26,),
        "explicit",
        "Created as the turn-by-turn summary for the session.",
    ),
]

DIR_REFS = [
    DirRef(
        "new/0-Context/Lessons",
        (2,),
        "Turn 2 referenced the directory as a set of three exemplar lesson files.",
    ),
    DirRef(
        "new/0-Context/MA-Course-Maps",
        (22,),
        "Turn 22 referenced the directory by name before specific maps were selected.",
    ),
]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(repo_rel_path: str, dest_root: Path) -> dict:
    src = REPO_ROOT / repo_rel_path
    dest = dest_root / repo_rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {
        "repo_path": repo_rel_path,
        "absolute_source": str(src),
        "bundle_copy": str(dest.relative_to(BUNDLE_ROOT)),
        "exists": src.exists(),
    }


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def parse_summary_turns(text: str) -> list[str]:
    matches = re.finditer(r"(?m)^(\d+)\.\s+(.*?)(?=^\d+\.\s+|\Z)", text, re.S)
    turns = []
    for match in matches:
        turns.append(re.sub(r"\s+\n", "\n", match.group(2).strip()))
    return turns


def load_session_prompts(session_id: str) -> list[dict]:
    prompts = []
    for line in CODEX_HISTORY.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("session_id") == session_id:
            prompts.append(obj)
    prompts.sort(key=lambda item: item["ts"])
    return prompts


def load_log_lines() -> list[str]:
    return CODEX_LOG.read_text(encoding="utf-8").splitlines()


def parse_iso_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def prompt_turn_for_timestamp(prompt_timestamps: list[float], timestamp: float) -> int | None:
    turn_number = None
    for index, prompt_ts in enumerate(prompt_timestamps, start=1):
        if prompt_ts <= timestamp:
            turn_number = index
        else:
            break
    return turn_number


def scope_end_prompt_index(prompts: list[dict]) -> int:
    for index, prompt in enumerate(prompts):
        if SCOPE_END_PROMPT_SNIPPET in prompt["text"]:
            return index
    raise RuntimeError("Could not find the summary-creation prompt in session history.")


def extract_apply_patches(lines: list[str], session_id: str) -> list[dict]:
    patches: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if session_id in line and "ToolCall: apply_patch " in line:
            submission_match = re.search(r'submission.id="([^"]+)"', line)
            timestamp = line.split(" ", 1)[0]
            patch_text = line.split("ToolCall: apply_patch ", 1)[1]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.startswith(" thread_id="):
                    break
                patch_text += "\n" + next_line
                i += 1
            file_paths = re.findall(
                r"(?m)^\*\*\* (?:Update|Add|Delete) File: (.+)$",
                patch_text,
            )
            patches.append(
                {
                    "timestamp": timestamp,
                    "submission_id": submission_match.group(1) if submission_match else None,
                    "patch_text": patch_text.rstrip(),
                    "file_paths": file_paths,
                }
            )
        i += 1
    return patches


def git_history_for(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--follow", "--date=iso", "--stat", "--patch", "--", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "<no git history output available>"


def dir_listing_for(path: str) -> str:
    full_path = REPO_ROOT / path
    entries = []
    for child in sorted(full_path.iterdir()):
        prefix = "[D]" if child.is_dir() else "[F]"
        entries.append(f"{prefix} {child.name}")
    return "\n".join(entries) + ("\n" if entries else "")


def build_readme(manifest: dict) -> str:
    return f"""# Conversation-Turn-History Bundle

This folder is a standalone analysis bundle for the session summarized in `Conversation Turn History.md`.

It includes:
- the turn summary itself,
- the exact user prompt stream up through the summary-creation step,
- current copies of every referenced file that could be identified,
- directory listings for referenced directories,
- recoverable `apply_patch` diffs extracted from the Codex log for the same session,
- Git history dumps for the referenced files.

Key limits:
- The exact user prompts are recoverable from `~/.codex/history.jsonl`.
- The exact assistant prose is not fully recoverable for every turn; the most reliable artifacts are the summary, the prompts, and the patch hunks.
- Some file references are marked as inferred when the summary referred to a directory rather than naming each file explicitly.
- The summary turns and the exact prompt stream are separated because they do not line up perfectly one-to-one late in the session.

Counts:
- Summary turns: {manifest["summary_turn_count"]}
- Exact prompts captured before the bundle cutoff: {manifest["prompt_count"]}
- Referenced files copied: {manifest["copied_file_count"]}
- Referenced directories listed: {manifest["directory_count"]}
- Recoverable `apply_patch` hunks before the bundle cutoff: {manifest["patch_count"]}

Suggested reading order for another AI:
1. `source/Conversation Turn History.md`
2. `session/turns-01-26.md`
3. `session/exact-prompts.md`
4. `patches/patch-index.json`
5. `files/current/`
6. `git-history/`
"""


def main() -> None:
    ensure_clean_dir(BUNDLE_ROOT / "source")
    ensure_clean_dir(BUNDLE_ROOT / "session")
    ensure_clean_dir(BUNDLE_ROOT / "patches")
    (BUNDLE_ROOT / "patches" / "by-prompt").mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(BUNDLE_ROOT / "files" / "current")
    ensure_clean_dir(BUNDLE_ROOT / "files" / "directory-listings")
    ensure_clean_dir(BUNDLE_ROOT / "git-history")
    ensure_clean_dir(BUNDLE_ROOT / "extras")

    shutil.copy2(SUMMARY_PATH, BUNDLE_ROOT / "source" / SUMMARY_PATH.name)
    if EXTRA_OUTPUT_11.exists():
        shutil.copy2(EXTRA_OUTPUT_11, BUNDLE_ROOT / "extras" / EXTRA_OUTPUT_11.name)

    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    summary_turns = parse_summary_turns(summary_text)
    prompts = load_session_prompts(SESSION_ID)
    end_prompt_index = scope_end_prompt_index(prompts)
    exact_prompts = prompts[: end_prompt_index + 1]
    cutoff_ts = float(prompts[end_prompt_index + 1]["ts"]) if len(prompts) > end_prompt_index + 1 else None
    log_lines = load_log_lines()
    prompt_timestamps = [float(item["ts"]) for item in exact_prompts]
    patches = []
    for patch in extract_apply_patches(log_lines, SESSION_ID):
        patch_ts = parse_iso_timestamp(patch["timestamp"])
        if cutoff_ts is not None and patch_ts >= cutoff_ts:
            continue
        prompt_number = prompt_turn_for_timestamp(prompt_timestamps, patch_ts)
        if prompt_number is None or prompt_number > len(exact_prompts):
            continue
        patch["prompt_number"] = prompt_number
        patches.append(patch)
    patches.sort(key=lambda item: (item["prompt_number"], item["timestamp"]))

    copied_files = []
    for ref in FILE_REFS:
        copied_files.append(
            {
                **copy_file(ref.path, BUNDLE_ROOT / "files" / "current"),
                "turns": list(ref.turns),
                "reference_type": ref.reference_type,
                "note": ref.note,
            }
        )

    directory_entries = []
    for ref in DIR_REFS:
        listing_text = dir_listing_for(ref.path)
        listing_name = sanitize(ref.path) + ".txt"
        listing_path = BUNDLE_ROOT / "files" / "directory-listings" / listing_name
        write_text(listing_path, listing_text)
        directory_entries.append(
            {
                "repo_path": ref.path,
                "turns": list(ref.turns),
                "note": ref.note,
                "bundle_copy": str(listing_path.relative_to(BUNDLE_ROOT)),
            }
        )

    git_history_entries = []
    for ref in FILE_REFS:
        history_text = git_history_for(ref.path)
        out_name = sanitize(ref.path) + ".patchlog.txt"
        out_path = BUNDLE_ROOT / "git-history" / out_name
        write_text(out_path, history_text + "\n")
        git_history_entries.append(
            {
                "repo_path": ref.path,
                "bundle_copy": str(out_path.relative_to(BUNDLE_ROOT)),
            }
        )

    patch_index = []
    patches_by_prompt: dict[int, list[dict]] = {}
    for patch_number, patch in enumerate(patches, start=1):
        prompt_number = patch["prompt_number"]
        per_prompt_dir = BUNDLE_ROOT / "patches" / "by-prompt" / f"prompt-{prompt_number:02d}"
        per_prompt_dir.mkdir(parents=True, exist_ok=True)
        patch_file = per_prompt_dir / f"{len(list(per_prompt_dir.glob('*.patch'))) + 1:02d}.patch"
        write_text(patch_file, patch["patch_text"] + "\n")
        entry = {
            "patch_number": patch_number,
            "prompt_number": prompt_number,
            "timestamp": patch["timestamp"],
            "submission_id": patch["submission_id"],
            "file_paths": patch["file_paths"],
            "bundle_copy": str(patch_file.relative_to(BUNDLE_ROOT)),
        }
        patch_index.append(entry)
        patches_by_prompt.setdefault(prompt_number, []).append(entry)

    write_text(
        BUNDLE_ROOT / "patches" / "patch-index.json",
        json.dumps(patch_index, indent=2) + "\n",
    )

    turns_payload = []
    turns_md_lines = [
        "# Turns 1-26",
        "",
        "Note",
        "- The summary text comes from `Conversation Turn History.md`.",
        "- This file is intentionally summary-first. The exact prompt stream is recorded separately in `session/exact-prompts.md`.",
        "- Because the summary and the prompt stream diverge late in the session, patches are indexed by prompt number rather than summary turn number.",
        "",
    ]
    for turn_number in range(1, TURN_LIMIT + 1):
        summary = summary_turns[turn_number - 1] if turn_number - 1 < len(summary_turns) else ""
        referenced_file_paths = [
            ref.path for ref in FILE_REFS if turn_number in ref.turns
        ] + [ref.path for ref in DIR_REFS if turn_number in ref.turns]
        payload = {
            "turn_number": turn_number,
            "summary": summary,
            "referenced_paths": referenced_file_paths,
        }
        turns_payload.append(payload)

        turns_md_lines.extend(
            [
                f"## Turn {turn_number}",
                "",
                "**Summary**",
                "",
                summary,
                "",
                "**Referenced Paths**",
                "",
            ]
        )
        if referenced_file_paths:
            turns_md_lines.extend([f"- `{path}`" for path in referenced_file_paths])
        else:
            turns_md_lines.append("- None")
        turns_md_lines.append("")

    write_text(
        BUNDLE_ROOT / "session" / "turns-01-26.json",
        json.dumps(turns_payload, indent=2) + "\n",
    )
    write_text(BUNDLE_ROOT / "session" / "turns-01-26.md", "\n".join(turns_md_lines))

    exact_prompt_payload = []
    exact_prompt_lines = [
        "# Exact Prompts",
        "",
        "Note",
        "- These are the exact user prompts recovered from `~/.codex/history.jsonl` up through the prompt that created `Conversation Turn History.md`.",
        "- Recoverable patches are attached to the most recent prompt before each patch timestamp.",
        "",
    ]
    for prompt_number, prompt in enumerate(exact_prompts, start=1):
        prompt_patch_entries = patches_by_prompt.get(prompt_number, [])
        exact_prompt_payload.append(
            {
                "prompt_number": prompt_number,
                "timestamp": prompt["ts"],
                "text": prompt["text"],
                "patches": prompt_patch_entries,
            }
        )
        exact_prompt_lines.extend(
            [
                f"## Prompt {prompt_number}",
                "",
                f"**Timestamp**: `{prompt['ts']}`",
                "",
                "**Text**",
                "",
                prompt["text"],
                "",
                "**Recoverable Patches**",
                "",
            ]
        )
        if prompt_patch_entries:
            for entry in prompt_patch_entries:
                exact_prompt_lines.append(
                    f"- `{entry['bundle_copy']}` -> {', '.join(entry['file_paths']) if entry['file_paths'] else 'no file path parsed'}"
                )
        else:
            exact_prompt_lines.append("- None")
        exact_prompt_lines.append("")

    write_text(BUNDLE_ROOT / "session" / "exact-prompts.md", "\n".join(exact_prompt_lines))
    write_text(
        BUNDLE_ROOT / "session" / "exact-prompts.json",
        json.dumps(exact_prompt_payload, indent=2) + "\n",
    )

    manifest = {
        "session_id": SESSION_ID,
        "summary_turn_count": len(summary_turns),
        "prompt_count": len(exact_prompts),
        "copied_file_count": len(copied_files),
        "directory_count": len(directory_entries),
        "patch_count": len(patch_index),
        "files": copied_files,
        "directories": directory_entries,
        "git_history": git_history_entries,
        "extras": [
            {
                "repo_path": "Output-11.md",
                "bundle_copy": "extras/Output-11.md",
                "note": "Included as an extra because it preserves the recovered prompt/response for turn 11.",
            }
        ]
        if EXTRA_OUTPUT_11.exists()
        else [],
    }
    write_text(BUNDLE_ROOT / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text(BUNDLE_ROOT / "README.md", build_readme(manifest))


if __name__ == "__main__":
    main()
