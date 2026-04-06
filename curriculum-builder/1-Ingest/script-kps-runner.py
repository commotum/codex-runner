#!/usr/bin/env python3
import argparse
import csv
import datetime
import hashlib
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RUNNER_VERSION = "2026-04-05-kps-runner-v1"
SEED_ROW_COUNT = 11
MAX_ENTRIES_PER_SPLIT = 5
CODEX_TIMEOUT_SECONDS = 1800
MAX_LOG_CHARS = 4000
LIST_ITEM_RE = re.compile(r"^\s*(?:[*+-]\s+|\d+\.\s+)")
SPLIT_FILE_RE = re.compile(r"^KPs-Split-(\d+)\.md$")

KPS_DIR = Path(__file__).resolve().parent
REPO_ROOT = KPS_DIR.parent
PROMPT_PATH = KPS_DIR / "prompt-kps-cleanup.md"
SCHEMA_PATH = KPS_DIR / "kps-output.schema.json"
CSV_PATH = KPS_DIR / "Kps.csv"
STATE_PATH = KPS_DIR / ".kps_runner_state.json"
LOG_PATH = KPS_DIR / "kps_runner.log"
OUTPUT_DIR = KPS_DIR / ".kps-cleaned-json"
CODEX_CMD = "codex"

LOG_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


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


def load_state():
    if not STATE_PATH.exists():
        return {"completed": {}}
    try:
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("state root is not an object")
        completed = payload.get("completed")
        if not isinstance(completed, dict):
            payload["completed"] = {}
        return payload
    except Exception as exc:
        log_event(f"state_load_failed path={STATE_PATH} error={exc}")
        return {"completed": {}}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_name(f".{STATE_PATH.name}.tmp.{uuid.uuid4().hex}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(temp_path, STATE_PATH)


def clean_stale_temp_files():
    if OUTPUT_DIR.exists():
        for path in OUTPUT_DIR.glob("*.tmp.*"):
            try:
                path.unlink()
                log_event(f"stale_temp_removed path={path}")
            except Exception as exc:
                log_event(f"stale_temp_remove_failed path={path} error={exc}")
    for path in KPS_DIR.glob(".Kps.csv.tmp.*"):
        try:
            path.unlink()
            log_event(f"stale_temp_removed path={path}")
        except Exception as exc:
            log_event(f"stale_temp_remove_failed path={path} error={exc}")


def discover_split_files():
    files = []
    for path in KPS_DIR.iterdir():
        match = SPLIT_FILE_RE.match(path.name)
        if not match:
            continue
        files.append((int(match.group(1)), path.resolve()))
    return [path for _, path in sorted(files, key=lambda item: item[0])]


def filter_split_files(paths, requested_sources):
    if not requested_sources:
        return paths
    requested = set(requested_sources)
    filtered = []
    for path in paths:
        relative = path.relative_to(KPS_DIR).as_posix()
        if path.name in requested or relative in requested:
            filtered.append(path)
    return filtered


def extract_list_items(path):
    items = []
    current = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if LIST_ITEM_RE.match(line):
                if current:
                    items.append("\n".join(current).strip())
                current = [line]
                continue
            if current:
                current.append(line)
        if current:
            items.append("\n".join(current).strip())
    return items


def load_seed_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"missing seed CSV: {CSV_PATH}")
    rows = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["title", "description"]:
            raise ValueError(f"unexpected CSV header in {CSV_PATH}")
        for row in reader:
            rows.append(
                {
                    "title": (row.get("title") or "").strip(),
                    "description": (row.get("description") or "").strip(),
                }
            )
            if len(rows) == SEED_ROW_COUNT:
                break
    if len(rows) != SEED_ROW_COUNT:
        raise ValueError(f"expected {SEED_ROW_COUNT} seed rows in {CSV_PATH}, found {len(rows)}")
    return rows


def seed_reference_csv(seed_rows):
    lines = ['"title","description"']
    for row in seed_rows:
        title = row["title"].replace('"', '""')
        description = row["description"].replace('"', '""')
        lines.append(f'"{title}","{description}"')
    return "\n".join(lines)


def build_prompt(source_path, expected_count, prompt_template, seed_reference_text):
    prompt = prompt_template
    replacements = {
        "[SOURCE_MD_ABS_PATH]": str(source_path.resolve()),
        "[EXPECTED_ENTRY_COUNT]": str(expected_count),
        "[TARGET_STYLE_REFERENCE_CSV]": seed_reference_text,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def build_runner_signature(prompt_template, seed_rows):
    signature_input = json.dumps(
        {
            "runner_version": RUNNER_VERSION,
            "prompt_sha256": sha256_text(prompt_template),
            "schema_sha256": sha256_file(SCHEMA_PATH),
            "seed_rows": seed_rows,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256_text(signature_input)


def build_codex_command(output_json_path):
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
            str(SCHEMA_PATH),
            "-o",
            str(output_json_path),
        ]
    )
    return base


def validate_output_json(output_json_path, expected_count):
    with Path(output_json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("output JSON root must be an object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("output JSON must contain an entries array")
    if len(entries) != expected_count:
        raise ValueError(f"expected {expected_count} entries, found {len(entries)}")
    if len(entries) > MAX_ENTRIES_PER_SPLIT:
        raise ValueError(f"entries exceed max split size: {len(entries)}")

    cleaned = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} is not an object")
        if set(entry.keys()) != {"title", "description"}:
            raise ValueError(f"entry {index} must contain only title and description")

        title = (entry.get("title") or "").strip()
        description = (entry.get("description") or "").strip()

        if not title:
            raise ValueError(f"entry {index} has an empty title")
        if not description:
            raise ValueError(f"entry {index} has an empty description")
        if "\n" in title or "\r" in title:
            raise ValueError(f"entry {index} title must be a single line")
        if "\n" in description or "\r" in description:
            raise ValueError(f"entry {index} description must be a single line")
        if LIST_ITEM_RE.match(title):
            raise ValueError(f"entry {index} title still contains a list marker")
        if "**" in title or "**" in description:
            raise ValueError(f"entry {index} still contains markdown bold markers")
        if title.endswith("."):
            raise ValueError(f"entry {index} title should not end with a period")
        if not description.endswith("."):
            raise ValueError(f"entry {index} description should end with a period")
        if not description[:1].islower():
            raise ValueError(f"entry {index} description should start with a lowercase action verb")

        cleaned.append({"title": title, "description": description})
    return cleaned


def final_output_path_for(source_path):
    return OUTPUT_DIR / f"{source_path.stem}.json"


def temp_output_path_for(source_path):
    return OUTPUT_DIR / f"{source_path.stem}.tmp.{uuid.uuid4().hex}.json"


def rebuild_csv(seed_rows, split_files):
    rows = list(seed_rows)
    for split_path in split_files:
        output_path = final_output_path_for(split_path)
        if not output_path.exists():
            continue
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for entry in payload.get("entries", []):
            rows.append(
                {
                    "title": (entry.get("title") or "").strip(),
                    "description": (entry.get("description") or "").strip(),
                }
            )

    temp_csv_path = KPS_DIR / f".{CSV_PATH.name}.tmp.{uuid.uuid4().hex}"
    with temp_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "description"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_csv_path, CSV_PATH)
    log_event(f"csv_rebuilt path={CSV_PATH} row_count={len(rows)}")


def should_skip(state, source_path, source_hash, runner_signature, overwrite):
    if overwrite:
        return False
    record = state.get("completed", {}).get(source_path.name)
    if not isinstance(record, dict):
        return False
    if record.get("source_hash") != source_hash:
        return False
    if record.get("runner_signature") != runner_signature:
        return False
    output_path = final_output_path_for(source_path)
    return output_path.exists()


def run_codex(prompt, output_json_path):
    args = build_codex_command(output_json_path)
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
        stdout, stderr = proc.communicate(input=prompt, timeout=CODEX_TIMEOUT_SECONDS)
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
        raise RuntimeError(f"codex timed out after {CODEX_TIMEOUT_SECONDS} seconds") from exc

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


def process_split(
    source_path,
    prompt_template,
    seed_reference_text,
    runner_signature,
    state,
    all_split_files,
    seed_rows,
    dry_run,
    overwrite,
):
    source_hash = sha256_file(source_path)
    items = extract_list_items(source_path)
    if not items:
        raise ValueError(f"no list items found in {source_path}")
    expected_count = len(items)

    with STATE_LOCK:
        if should_skip(state, source_path, source_hash, runner_signature, overwrite):
            log_event(f"file_skipped path={source_path} reason=up_to_date")
            return "skipped"

    prompt = build_prompt(source_path, expected_count, prompt_template, seed_reference_text)
    output_path = final_output_path_for(source_path)
    temp_output_path = temp_output_path_for(source_path)

    if dry_run:
        log_event(
            f"dry_run path={source_path} expected_count={expected_count} output={output_path}"
        )
        return "dry_run"

    log_event(
        f"file_start path={source_path} expected_count={expected_count} temp_output={temp_output_path}"
    )
    try:
        run_codex(prompt, temp_output_path)
        cleaned_entries = validate_output_json(temp_output_path, expected_count)
        final_payload = {"entries": cleaned_entries}
        normalized_temp = temp_output_path_for(source_path)
        with normalized_temp.open("w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=2, ensure_ascii=False)
        os.replace(normalized_temp, output_path)
    finally:
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                pass

    with STATE_LOCK:
        state.setdefault("completed", {})[source_path.name] = {
            "completed_at": now_iso(),
            "expected_count": expected_count,
            "output_json": str(output_path),
            "runner_signature": runner_signature,
            "source_hash": source_hash,
        }
        save_state(state)
        rebuild_csv(seed_rows=seed_rows, split_files=all_split_files)

    log_event(f"file_success path={source_path} output={output_path}")
    return "processed"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Codex over KPs split files and rebuild Kps.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Build the queue and prompts without calling Codex.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess split files even when state says they are up to date.")
    parser.add_argument("--workers", type=int, default=1, help="Number of split files to process in parallel.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many queued split files.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Limit processing to specific split files by basename or KPs-relative path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if shutil.which(shlex.split(CODEX_CMD)[0]) is None:
        raise SystemExit(f"missing Codex CLI: {CODEX_CMD}")

    clean_stale_temp_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_split_files = discover_split_files()
    split_files = filter_split_files(all_split_files, args.source)
    if args.limit > 0:
        split_files = split_files[: args.limit]
    if not split_files:
        raise SystemExit("no split files matched the requested filters")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    seed_rows = load_seed_rows()
    seed_reference_text = seed_reference_csv(seed_rows)
    runner_signature = build_runner_signature(prompt_template, seed_rows)
    state = load_state()

    log_event(
        f"run_start split_count={len(split_files)} dry_run={args.dry_run} overwrite={args.overwrite} workers={args.workers}"
    )

    processed = 0
    skipped = 0
    dry_run_count = 0

    worker_count = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_split,
                source_path,
                prompt_template,
                seed_reference_text,
                runner_signature,
                state,
                all_split_files,
                seed_rows,
                args.dry_run,
                args.overwrite,
            ): source_path
            for source_path in split_files
        }
        for future in as_completed(futures):
            source_path = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                log_event(f"file_failed path={source_path} error={exc}")
                raise
            if status == "processed":
                processed += 1
            elif status == "skipped":
                skipped += 1
            elif status == "dry_run":
                dry_run_count += 1

    if not args.dry_run:
        rebuild_csv(seed_rows=seed_rows, split_files=all_split_files)

    log_event(
        f"run_complete processed={processed} skipped={skipped} dry_run={dry_run_count} split_count={len(split_files)}"
    )
    print(
        json.dumps(
            {
                "processed": processed,
                "skipped": skipped,
                "dry_run": dry_run_count,
                "split_count": len(split_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
