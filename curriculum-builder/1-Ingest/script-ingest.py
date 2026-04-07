#!/usr/bin/env python3
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


RUNNER_VERSION = "2026-04-06-ingest-v1"
MAX_LOG_CHARS = 4000
COURSE_ID_RE = re.compile(r"^[A-Z0-9]{3}$")
DOCUMENT_INDEX_RE = re.compile(r"^\d{4}$")
WORD_RE = re.compile(r"[A-Za-z0-9]+")

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
SOURCE_ROOT_DEFAULT = PIPELINE_ROOT / "0-Source"
COURSES_CSV_DEFAULT = SOURCE_ROOT_DEFAULT / "Courses.csv"
PROMPTS_DIR = STAGE_DIR / "Prompts"
COURSE_NAME_PROMPT_PATH = PROMPTS_DIR / "course-name.md"
COURSE_NAME_SCHEMA_PATH = PROMPTS_DIR / "course-name.schema.json"
COURSE_NAME_OUTPUT_DIR = STAGE_DIR / ".course-name-json"
LECTURE_TOPICS_PROMPT_PATH = PROMPTS_DIR / "lecture-topics.md"
LECTURE_TOPICS_SCHEMA_PATH = PROMPTS_DIR / "lecture-topics.schema.json"
LECTURE_TOPICS_OUTPUT_DIR = STAGE_DIR / ".lecture-topics-json"
STATE_PATH = STAGE_DIR / ".ingest_state.json"
LOG_PATH = STAGE_DIR / "ingest.log"
TARGETS_PATH = STAGE_DIR / "ingest-targets.json"
CODEX_CMD = "codex"
TOPICS_DIRNAME = "Topics"
TOPICS_CSV_FIELDNAMES = ["tile", "description"]

LOG_LOCK = threading.Lock()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def progress_enabled():
    return tqdm is not None and sys.stderr.isatty()


class NullProgress:
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix_str(self, text, refresh=True):
        return None

    def update(self, n=1):
        return None

    def close(self):
        return None


def make_progress(iterable, desc, unit):
    if not progress_enabled():
        return NullProgress(iterable)
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True)


def log_event(message):
    line = f"{now_iso()} {message}\n"
    with LOG_LOCK:
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except Exception:
            sys.stderr.write(line)


def sanitize_log_text(text, max_chars=MAX_LOG_CHARS):
    if not text:
        return ""
    scrubbed = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(scrubbed) <= max_chars:
        return scrubbed
    return scrubbed[:max_chars] + "...(truncated)"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt_signature(prompt_path, schema_path):
    payload = json.dumps(
        {
            "runner_version": RUNNER_VERSION,
            "prompt_sha256": sha256_file(prompt_path),
            "schema_sha256": sha256_file(schema_path),
        },
        sort_keys=True,
    )
    return sha256_text(payload)


def load_state():
    if not STATE_PATH.exists():
        return {
            "registered_courses": {},
            "indexed_courses": {},
            "generated_topics": {},
            "last_targets": {},
        }
    try:
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("state root is not an object")
        payload.setdefault("registered_courses", {})
        payload.setdefault("indexed_courses", {})
        payload.setdefault("generated_topics", {})
        payload.setdefault("last_targets", {})
        return payload
    except Exception as exc:
        log_event(f"state_load_failed path={STATE_PATH} error={exc}")
        return {
            "registered_courses": {},
            "indexed_courses": {},
            "generated_topics": {},
            "last_targets": {},
        }


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_name(f".{STATE_PATH.name}.tmp.{uuid.uuid4().hex}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(temp_path, STATE_PATH)


def clean_stale_temp_files(source_root):
    for output_dir in (COURSE_NAME_OUTPUT_DIR, LECTURE_TOPICS_OUTPUT_DIR):
        if output_dir.exists():
            for path in output_dir.glob("*.tmp.*"):
                try:
                    path.unlink()
                    log_event(f"stale_temp_removed path={path}")
                except Exception as exc:
                    log_event(f"stale_temp_remove_failed path={path} error={exc}")
    for pattern in (".ingest_state.json.tmp.*", ".ingest-targets.json.tmp.*"):
        for path in STAGE_DIR.glob(pattern):
            try:
                path.unlink()
                log_event(f"stale_temp_removed path={path}")
            except Exception as exc:
                log_event(f"stale_temp_remove_failed path={path} error={exc}")
    if source_root.exists():
        for path in source_root.rglob(".*.tmp.*"):
            if not path.is_file():
                continue
            try:
                path.unlink()
                log_event(f"stale_temp_removed path={path}")
            except Exception as exc:
                log_event(f"stale_temp_remove_failed path={path} error={exc}")


def repo_relative(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return str(path)


def slugify_text(text):
    parts = WORD_RE.findall(text)
    return "-".join(part.lower() for part in parts)


def read_csv_rows(path, expected_fieldnames):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fieldnames:
            raise ValueError(f"unexpected CSV header in {path}: {reader.fieldnames}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def write_csv_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def render_csv_text(fieldnames, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().strip()


def ensure_codex_available():
    executable = shlex.split(CODEX_CMD)[0]
    if shutil.which(executable) is None:
        raise SystemExit(f"missing Codex CLI: {CODEX_CMD}")


def ensure_courses_csv(path, dry_run):
    if path.exists():
        return
    if dry_run:
        log_event(f"dry_run_courses_csv_create path={path}")
        return
    write_csv_rows(path, ["course-id", "course-name"], [])
    log_event(f"courses_csv_created path={path}")


def load_courses_index(path):
    rows = read_csv_rows(path, ["course-id", "course-name"])
    seen_ids = set()
    seen_slugs = set()
    for row in rows:
        course_id = row["course-id"]
        course_name = row["course-name"]
        slug = slugify_text(course_name)
        if not COURSE_ID_RE.match(course_id):
            raise ValueError(f"invalid course-id in {path}: {course_id}")
        if not course_name:
            raise ValueError(f"empty course-name in {path}")
        if course_id in seen_ids:
            raise ValueError(f"duplicate course-id in {path}: {course_id}")
        if slug in seen_slugs:
            raise ValueError(f"duplicate normalized course-name in {path}: {course_name}")
        seen_ids.add(course_id)
        seen_slugs.add(slug)
    return rows


def courses_by_slug(rows):
    return {slugify_text(row["course-name"]): row for row in rows}


def discover_course_dirs(source_root):
    course_dirs = []
    for path in source_root.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        course_dirs.append(path.resolve())
    return sorted(course_dirs, key=lambda item: item.name.lower())


def build_course_name_prompt(prompt_template, courses_csv_path, folder_basename):
    prompt = prompt_template
    replacements = {
        "[COURSES_CSV_ABS_PATH]": str(courses_csv_path.resolve()),
        "[COURSE_FOLDER_BASENAME]": folder_basename,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def build_lecture_topics_prompt(prompt_template, source_md_path, topics_csv_path):
    prompt = prompt_template
    replacements = {
        "[SOURCE_MD_ABS_PATH]": str(source_md_path.resolve()),
        "[TOPICS_CSV_ABS_PATH]": str(topics_csv_path.resolve()),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def detect_document_kind(document_path):
    normalized_path = document_path.replace("\\", "/")
    if normalized_path.startswith("Lectures/") and normalized_path.lower().endswith(".md"):
        return "lecture"
    return "unknown"


def build_codex_command(schema_path, output_path):
    base = shlex.split(CODEX_CMD)
    if not base:
        raise ValueError("empty CODEX_CMD")
    base.extend(
        [
            "exec",
            "-s",
            "read-only",
            "-C",
            str(REPO_ROOT),
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ]
    )
    return base


def run_codex(prompt, schema_path, output_path):
    args = build_codex_command(schema_path, output_path)
    cmd_display = " ".join(shlex.quote(arg) for arg in args)
    log_event(f"codex_launch cmd={cmd_display}")
    start_time = time.monotonic()
    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to launch Codex: {exc}") from exc

    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - start_time
        log_event(
            "codex_timeout "
            f"seconds={elapsed:.1f} "
            f"stdout={sanitize_log_text(stdout)} "
            f"stderr={sanitize_log_text(stderr)}"
        )
        raise RuntimeError("codex timed out after 1800 seconds") from exc

    elapsed = time.monotonic() - start_time
    log_event(
        "codex_result "
        f"rc={proc.returncode} "
        f"seconds={elapsed:.1f} "
        f"stdout={sanitize_log_text(stdout)} "
        f"stderr={sanitize_log_text(stderr)}"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"codex exited with rc={proc.returncode}")


def validate_course_output(output_path, folder_basename, course_rows):
    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("course output root must be an object")
    if set(payload.keys()) != {"course_id", "course_name"}:
        raise ValueError("course output must contain only course_id and course_name")

    course_id = (payload.get("course_id") or "").strip()
    course_name = (payload.get("course_name") or "").strip()
    if not COURSE_ID_RE.match(course_id):
        raise ValueError(f"invalid course_id: {course_id}")
    if not course_name:
        raise ValueError("course_name cannot be empty")
    if slugify_text(course_name) != slugify_text(folder_basename):
        raise ValueError(
            f"course_name does not normalize back to folder basename: {course_name} vs {folder_basename}"
        )

    used_ids = {row["course-id"] for row in course_rows}
    used_slugs = {slugify_text(row["course-name"]) for row in course_rows}
    if course_id in used_ids:
        raise ValueError(f"course_id already exists in Courses.csv: {course_id}")
    if slugify_text(course_name) in used_slugs:
        raise ValueError(f"course_name already exists in Courses.csv: {course_name}")

    return {"course-id": course_id, "course-name": course_name}


def validate_topics_output(output_path):
    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("topics output root must be an object")
    if set(payload.keys()) != {"entries"}:
        raise ValueError("topics output must contain only the entries field")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("topics entries must be a non-empty array")

    normalized_entries = []
    seen_titles = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"topics entry {index} must be an object")
        if set(entry.keys()) != {"title", "description"}:
            raise ValueError(f"topics entry {index} must contain only title and description")

        title = (entry.get("title") or "").strip()
        description = (entry.get("description") or "").strip()
        if not title:
            raise ValueError(f"topics entry {index} has empty title")
        if not description:
            raise ValueError(f"topics entry {index} has empty description")
        if "\n" in title or "\r" in title:
            raise ValueError(f"topics entry {index} title must be single-line")
        if "\n" in description or "\r" in description:
            raise ValueError(f"topics entry {index} description must be single-line")
        if title.endswith("."):
            raise ValueError(f"topics entry {index} title must not end with a period")
        if not description.endswith("."):
            raise ValueError(f"topics entry {index} description must end with a period")

        normalized_title = title.casefold()
        if normalized_title in seen_titles:
            raise ValueError(f"duplicate topics title: {title}")
        seen_titles.add(normalized_title)
        normalized_entries.append({"title": title, "description": description})

    return {"entries": normalized_entries}


def build_topics_csv_rows(entries):
    return [{"tile": entry["title"], "description": entry["description"]} for entry in entries]


def write_topics_csv(path, entries):
    write_csv_rows(path, TOPICS_CSV_FIELDNAMES, build_topics_csv_rows(entries))


def register_new_course(course_dir, courses_csv_path, prompt_template, course_rows, prompt_signature, state, dry_run):
    prompt = build_course_name_prompt(prompt_template, courses_csv_path, course_dir.name)
    output_path = COURSE_NAME_OUTPUT_DIR / f"{course_dir.name}.json"
    temp_output_path = COURSE_NAME_OUTPUT_DIR / f"{course_dir.name}.tmp.{uuid.uuid4().hex}.json"

    if dry_run:
        log_event(f"dry_run_course_register folder={course_dir.name} output={output_path}")
        return None

    COURSE_NAME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_codex(prompt, COURSE_NAME_SCHEMA_PATH, temp_output_path)
    course_row = validate_course_output(temp_output_path, course_dir.name, course_rows)
    normalized_payload = {
        "course_id": course_row["course-id"],
        "course_name": course_row["course-name"],
        "source_folder": course_dir.name,
    }
    write_json(output_path, normalized_payload)
    if temp_output_path.exists():
        temp_output_path.unlink()

    updated_rows = list(course_rows)
    updated_rows.append(course_row)
    write_csv_rows(courses_csv_path, ["course-id", "course-name"], updated_rows)
    course_rows[:] = updated_rows

    state["registered_courses"][course_dir.name] = {
        "completed_at": now_iso(),
        "course_id": course_row["course-id"],
        "course_name": course_row["course-name"],
        "output_json": repo_relative(output_path),
        "prompt_signature": prompt_signature,
    }
    save_state(state)
    log_event(
        f"course_registered folder={course_dir.name} course_id={course_row['course-id']} course_name={course_row['course-name']}"
    )
    return course_row


def discover_source_documents(course_root):
    documents = []
    for path in course_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(course_root)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        if TOPICS_DIRNAME in rel_path.parts:
            continue
        if len(rel_path.parts) == 1 and rel_path.name.endswith("-Contents.csv"):
            continue
        documents.append(rel_path.as_posix())
    return sorted(documents)


def load_contents_index(path, course_id):
    rows = read_csv_rows(path, ["index", "document-id", "document-path", "document-ingested"])
    seen_indices = set()
    seen_paths = set()
    for row in rows:
        index_value = row["index"]
        document_id = row["document-id"]
        document_path = row["document-path"]
        if not DOCUMENT_INDEX_RE.match(index_value):
            raise ValueError(f"invalid index in {path}: {index_value}")
        expected_id = f"{course_id}-SRC-{index_value}"
        if document_id != expected_id:
            raise ValueError(f"unexpected document-id in {path}: {document_id} != {expected_id}")
        if not document_path:
            raise ValueError(f"empty document-path in {path}")
        if index_value in seen_indices:
            raise ValueError(f"duplicate index in {path}: {index_value}")
        if document_path in seen_paths:
            raise ValueError(f"duplicate document-path in {path}: {document_path}")
        seen_indices.add(index_value)
        seen_paths.add(document_path)
    return rows


def sync_contents_index(course_dir, course_row, state, dry_run):
    course_id = course_row["course-id"]
    contents_path = course_dir / f"{course_id}-Contents.csv"
    existing_rows = load_contents_index(contents_path, course_id)
    discovered_paths = discover_source_documents(course_dir)
    existing_by_path = {row["document-path"]: row for row in existing_rows}
    max_index = 0
    for row in existing_rows:
        max_index = max(max_index, int(row["index"]))

    new_rows = []
    for document_path in discovered_paths:
        if document_path in existing_by_path:
            continue
        max_index += 1
        if max_index > 9999:
            raise ValueError(f"contents index exceeded 9999 rows for {course_dir}")
        index_value = f"{max_index:04d}"
        new_rows.append(
            {
                "index": index_value,
                "document-id": f"{course_id}-SRC-{index_value}",
                "document-path": document_path,
                "document-ingested": "",
            }
        )

    stale_paths = sorted(set(existing_by_path) - set(discovered_paths))
    final_rows = list(existing_rows) + new_rows

    if dry_run:
        log_event(
            f"dry_run_contents_sync course_id={course_id} path={contents_path} existing={len(existing_rows)} new={len(new_rows)} stale={len(stale_paths)}"
        )
    else:
        if not contents_path.exists() or new_rows:
            write_csv_rows(
                contents_path,
                ["index", "document-id", "document-path", "document-ingested"],
                final_rows,
            )
            log_event(
                f"contents_index_written course_id={course_id} path={contents_path} row_count={len(final_rows)} added={len(new_rows)}"
            )
        else:
            log_event(
                f"contents_index_up_to_date course_id={course_id} path={contents_path} row_count={len(final_rows)}"
            )
        state["indexed_courses"][course_id] = {
            "completed_at": now_iso(),
            "contents_index": repo_relative(contents_path),
            "row_count": len(final_rows),
            "added_count": len(new_rows),
            "stale_count": len(stale_paths),
        }
        save_state(state)

    if stale_paths:
        log_event(
            f"contents_index_stale_paths course_id={course_id} count={len(stale_paths)} paths={sanitize_log_text(json.dumps(stale_paths))}"
        )

    return {
        "course_id": course_id,
        "course_name": course_row["course-name"],
        "course_dir": course_dir,
        "contents_path": contents_path,
        "rows": final_rows,
        "new_count": len(new_rows),
        "stale_count": len(stale_paths),
    }


def build_topics_csv_path(course_dir, document_id):
    return course_dir / TOPICS_DIRNAME / f"{document_id}-Topics.csv"


def build_topics_json_path(document_id):
    return LECTURE_TOPICS_OUTPUT_DIR / f"{document_id}.json"


def stamp_document_ingested(contents_path, course_id, document_id, ingested_at):
    rows = load_contents_index(contents_path, course_id)
    updated = False
    for row in rows:
        if row["document-id"] == document_id:
            row["document-ingested"] = ingested_at
            updated = True
            break
    if not updated:
        raise ValueError(f"document-id not found in {contents_path}: {document_id}")
    write_csv_rows(
        contents_path,
        ["index", "document-id", "document-path", "document-ingested"],
        rows,
    )


def build_targets_manifest(source_root, indexed_courses, include_ingested=False):
    targets = []
    progress = make_progress(indexed_courses, desc="Phase 3/4 Acquiring targets", unit="course")
    for entry in progress:
        progress.set_postfix_str(entry["course_id"])
        course_dir = entry["course_dir"]
        contents_path = entry["contents_path"]
        for row in entry["rows"]:
            if not include_ingested and (row["document-ingested"] or "").strip():
                continue
            source_path = course_dir / row["document-path"]
            if not source_path.exists():
                log_event(
                    f"target_missing_source course_id={entry['course_id']} document_id={row['document-id']} path={source_path}"
                )
                continue
            targets.append(
                {
                    "course_id": entry["course_id"],
                    "course_name": entry["course_name"],
                    "course_root": repo_relative(course_dir),
                    "contents_index": repo_relative(contents_path),
                    "document_index": row["index"],
                    "document_id": row["document-id"],
                    "document_kind": detect_document_kind(row["document-path"]),
                    "document_path": row["document-path"],
                    "document_ingested": row["document-ingested"],
                    "source_path": repo_relative(source_path),
                }
            )
    progress.close()

    return {
        "generated_at": now_iso(),
        "runner_version": RUNNER_VERSION,
        "source_root": repo_relative(source_root),
        "target_count": len(targets),
        "targets": targets,
    }


def target_matches_document_filter(target, requested):
    requested_lower = requested.strip().lower()
    if not requested_lower:
        return False
    candidates = {
        target["document_id"],
        target["document_path"],
        target["source_path"],
        Path(target["document_path"]).name,
        Path(target["document_path"]).stem,
    }
    return any(requested_lower in candidate.lower() for candidate in candidates)


def select_targets(targets, requested_documents):
    if not requested_documents:
        return list(targets)
    selected = []
    for target in targets:
        if any(target_matches_document_filter(target, requested) for requested in requested_documents):
            selected.append(target)
    return selected


def generate_topics_for_target(
    target,
    lecture_prompt_template,
    lecture_prompt_signature,
    state,
    dry_run,
):
    if target["document_kind"] != "lecture":
        log_event(
            f"target_topic_generation_skipped document_id={target['document_id']} reason=unsupported_kind kind={target['document_kind']}"
        )
        return {"status": "skipped", "document_id": target["document_id"], "entry_count": 0}

    course_dir = REPO_ROOT / target["course_root"]
    contents_path = REPO_ROOT / target["contents_index"]
    source_path = REPO_ROOT / target["source_path"]
    topics_csv_path = build_topics_csv_path(course_dir, target["document_id"])
    topics_json_path = build_topics_json_path(target["document_id"])
    temp_output_path = topics_json_path.with_name(
        f"{topics_json_path.name}.tmp.{uuid.uuid4().hex}"
    )
    prompt = build_lecture_topics_prompt(
        lecture_prompt_template,
        source_path,
        topics_csv_path,
    )

    if dry_run:
        log_event(
            "dry_run_topic_generation "
            f"document_id={target['document_id']} "
            f"source={source_path} "
            f"topics_csv={topics_csv_path} "
            f"topics_json={topics_json_path}"
        )
        return {"status": "dry-run", "document_id": target["document_id"], "entry_count": 0}

    LECTURE_TOPICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_codex(prompt, LECTURE_TOPICS_SCHEMA_PATH, temp_output_path)
    validated_payload = validate_topics_output(temp_output_path)
    write_json(topics_json_path, validated_payload)
    write_topics_csv(topics_csv_path, validated_payload["entries"])

    completed_at = now_iso()
    stamp_document_ingested(
        contents_path,
        target["course_id"],
        target["document_id"],
        completed_at,
    )
    state["generated_topics"][target["document_id"]] = {
        "completed_at": completed_at,
        "course_id": target["course_id"],
        "document_kind": target["document_kind"],
        "entry_count": len(validated_payload["entries"]),
        "output_csv": repo_relative(topics_csv_path),
        "output_json": repo_relative(topics_json_path),
        "prompt_signature": lecture_prompt_signature,
        "source_path": target["source_path"],
    }
    save_state(state)
    log_event(
        "topics_generated "
        f"document_id={target['document_id']} "
        f"entry_count={len(validated_payload['entries'])} "
        f"topics_csv={topics_csv_path}"
    )
    if temp_output_path.exists():
        temp_output_path.unlink()
    return {
        "status": "generated",
        "document_id": target["document_id"],
        "entry_count": len(validated_payload["entries"]),
    }


def select_course_dirs(course_dirs, course_rows, requested):
    if not requested:
        return course_dirs
    requested_set = set(requested)
    requested_slugs = {slugify_text(item) for item in requested}
    by_slug = courses_by_slug(course_rows)
    selected = []
    for course_dir in course_dirs:
        folder_slug = slugify_text(course_dir.name)
        row = by_slug.get(folder_slug)
        course_id = row["course-id"] if row else ""
        course_name = row["course-name"] if row else ""
        if (
            course_dir.name in requested_set
            or folder_slug in requested_slugs
            or course_id in requested_set
            or course_name in requested_set
            or slugify_text(course_name) in requested_slugs
        ):
            selected.append(course_dir)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Source management, target acquisition, and topic generation for curriculum-builder ingestion."
    )
    parser.add_argument(
        "--source-root",
        default=str(SOURCE_ROOT_DEFAULT),
        help="Top-level source root to scan for courses.",
    )
    parser.add_argument(
        "--courses-csv",
        default=str(COURSES_CSV_DEFAULT),
        help="Path to the course index CSV.",
    )
    parser.add_argument(
        "--course",
        action="append",
        default=[],
        help="Limit processing to specific course folders, slugs, ids, or names.",
    )
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        help="Limit topic generation to specific document ids, basenames, or paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate topics for documents even if document-ingested is already set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and validate work without writing files or launching Codex.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    courses_csv_path = Path(args.courses_csv).resolve()
    if not source_root.exists():
        raise SystemExit(f"missing source root: {source_root}")
    if courses_csv_path.parent != source_root:
        log_event(
            f"courses_csv_outside_source_root path={courses_csv_path} source_root={source_root}"
        )

    clean_stale_temp_files(source_root)
    ensure_courses_csv(courses_csv_path, args.dry_run)

    course_prompt_template = COURSE_NAME_PROMPT_PATH.read_text(encoding="utf-8")
    course_prompt_signature = build_prompt_signature(
        COURSE_NAME_PROMPT_PATH, COURSE_NAME_SCHEMA_PATH
    )
    lecture_prompt_template = LECTURE_TOPICS_PROMPT_PATH.read_text(encoding="utf-8")
    lecture_prompt_signature = build_prompt_signature(
        LECTURE_TOPICS_PROMPT_PATH, LECTURE_TOPICS_SCHEMA_PATH
    )
    state = load_state()

    course_dirs = discover_course_dirs(source_root)
    course_rows = load_courses_index(courses_csv_path)
    selected_course_dirs = select_course_dirs(course_dirs, course_rows, args.course)
    if args.course and not selected_course_dirs:
        raise SystemExit("no course folders matched the requested filters")

    slug_map = courses_by_slug(course_rows)
    new_course_dirs = [
        course_dir for course_dir in selected_course_dirs if slugify_text(course_dir.name) not in slug_map
    ]

    log_event(
        "run_start "
        f"source_root={source_root} "
        f"course_dir_count={len(course_dirs)} "
        f"selected_course_count={len(selected_course_dirs)} "
        f"new_course_count={len(new_course_dirs)} "
        f"document_filter_count={len(args.document)} "
        f"overwrite={args.overwrite} "
        f"dry_run={args.dry_run}"
    )

    if new_course_dirs and not args.dry_run:
        ensure_codex_available()
        progress = make_progress(new_course_dirs, desc="Phase 1/4 Registering courses", unit="course")
        for course_dir in progress:
            progress.set_postfix_str(course_dir.name)
            register_new_course(
                course_dir,
                courses_csv_path,
                course_prompt_template,
                course_rows,
                course_prompt_signature,
                state,
                args.dry_run,
            )
        progress.close()
        slug_map = courses_by_slug(course_rows)
    elif new_course_dirs:
        progress = make_progress(new_course_dirs, desc="Phase 1/4 Registering courses", unit="course")
        for course_dir in progress:
            progress.set_postfix_str(course_dir.name)
            log_event(f"dry_run_new_course_pending folder={course_dir.name}")
        progress.close()

    indexed_courses = []
    progress = make_progress(selected_course_dirs, desc="Phase 2/4 Syncing contents", unit="course")
    for course_dir in progress:
        progress.set_postfix_str(course_dir.name)
        course_row = slug_map.get(slugify_text(course_dir.name))
        if not course_row:
            log_event(
                f"course_unregistered_skipped folder={course_dir.name} reason=missing_courses_csv_entry"
            )
            continue
        indexed_courses.append(sync_contents_index(course_dir, course_row, state, args.dry_run))
    progress.close()

    manifest = build_targets_manifest(
        source_root,
        indexed_courses,
        include_ingested=args.overwrite,
    )
    manifest["targets"] = select_targets(manifest["targets"], args.document)
    manifest["target_count"] = len(manifest["targets"])
    if args.document and not manifest["targets"]:
        raise SystemExit("no documents matched the requested filters")

    if args.dry_run:
        log_event(
            f"dry_run_targets_ready target_count={manifest['target_count']} selected_courses={len(indexed_courses)}"
        )
    else:
        write_json(TARGETS_PATH, manifest)
        state["last_targets"] = {
            "generated_at": manifest["generated_at"],
            "path": repo_relative(TARGETS_PATH),
            "target_count": manifest["target_count"],
        }
        save_state(state)
        log_event(
            f"targets_manifest_written path={TARGETS_PATH} target_count={manifest['target_count']}"
        )

    topics_generated = 0
    topics_skipped = 0
    progress = make_progress(manifest["targets"], desc="Phase 4/4 Generating topics", unit="doc")
    if manifest["targets"] and not args.dry_run:
        ensure_codex_available()
    for target in progress:
        progress.set_postfix_str(target["document_id"])
        result = generate_topics_for_target(
            target,
            lecture_prompt_template,
            lecture_prompt_signature,
            state,
            args.dry_run,
        )
        if result["status"] == "generated":
            topics_generated += 1
        elif result["status"] == "skipped":
            topics_skipped += 1
    progress.close()

    print(
        json.dumps(
            {
                "courses_discovered": len(course_dirs),
                "courses_selected": len(selected_course_dirs),
                "new_courses": len(new_course_dirs),
                "courses_indexed": len(indexed_courses),
                "target_count": manifest["target_count"],
                "topics_generated": topics_generated,
                "topics_skipped": topics_skipped,
                "dry_run": args.dry_run,
                "targets_manifest": repo_relative(TARGETS_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )
    log_event(
        "run_complete "
        f"selected_courses={len(indexed_courses)} "
        f"target_count={manifest['target_count']} "
        f"topics_generated={topics_generated} "
        f"topics_skipped={topics_skipped} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
