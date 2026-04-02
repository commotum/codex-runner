import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid


BOOK_ROOT = "/home/jake/Developer/codex-runner/The Math Academy Way"
PROMPT_PATH = "/home/jake/Developer/codex-runner/MA-Runner/prompt-ma-cleanup.md"
INVENTORY_PATH = "/home/jake/Developer/codex-runner/MA-Runner/math-academy-inventory.json"
STATE_PATH = "/home/jake/Developer/codex-runner/MA-Runner/.ma_cleanup_state.json"
LOG_PATH = "/home/jake/Developer/codex-runner/MA-Runner/ma_cleanup.log"
CODEX_CLI_CMD = "codex"
CODEX_EXEC_ARGS = ["exec", "--full-auto"]
CODEX_EXEC_TIMEOUT = 7200
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
MAX_LOG_CHARS = 4000
MIN_MD_BYTES = 20
DRY_RUN = False
OVERWRITE = False
SORT_MODE = "alpha"
CLEANUP_TEMP_ON_START = True


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path, message):
    line = f"{now_iso()} {message}\n"
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        sys.stderr.write(line)


def sanitize_log_text(text, max_chars):
    if not text:
        return ""
    scrubbed = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(scrubbed) <= max_chars:
        return scrubbed
    return scrubbed[:max_chars] + "...(truncated)"


def find_git_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_state(state_path, log_path):
    if not os.path.exists(state_path):
        return {"completed": {}}
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("state is not a dict")
        completed = data.get("completed")
        if not isinstance(completed, dict):
            data["completed"] = {}
        return data
    except Exception as exc:
        log_event(log_path, f"state_load_failed path={state_path} error={exc}")
        return {"completed": {}}


def save_state(state_path, state, log_path):
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except Exception as exc:
        log_event(log_path, f"state_save_failed path={state_path} error={exc}")


def extract_title(path):
    heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                match = heading_pattern.match(line)
                if match:
                    return match.group(1).strip()
    except Exception:
        return None
    return None


def list_markdown_files(book_root):
    files = []
    for root, _, names in os.walk(book_root):
        for name in names:
            if not name.lower().endswith(".md"):
                continue
            files.append(os.path.join(root, name))
    return files


def sort_queue(entries, sort_mode, log_path):
    mode = (sort_mode or "").strip().lower()
    if mode == "alpha":
        return sorted(entries, key=lambda path: os.path.relpath(path).lower())
    if mode in ("mtime", "mtime-desc", "newest"):
        return sorted(entries, key=os.path.getmtime, reverse=True)
    if mode in ("mtime-asc", "oldest"):
        return sorted(entries, key=os.path.getmtime)
    log_event(log_path, f"unknown_sort_mode mode={sort_mode} fallback=alpha")
    return sorted(entries, key=lambda path: os.path.relpath(path).lower())


def build_inventory(book_root, inventory_path, log_path):
    entries = []
    for path in sort_queue(list_markdown_files(book_root), "alpha", log_path):
        relative_path = os.path.relpath(path, book_root)
        entries.append(
            {
                "title": extract_title(path),
                "relative_path": relative_path.replace(os.sep, "/"),
                "absolute_path": os.path.abspath(path),
                "directory": os.path.dirname(relative_path).replace(os.sep, "/"),
                "file_name": os.path.basename(path),
            }
        )

    payload = {
        "generated_at": now_iso(),
        "book_root": os.path.abspath(book_root),
        "documents": entries,
    }
    os.makedirs(os.path.dirname(inventory_path), exist_ok=True)
    with open(inventory_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    log_event(log_path, f"inventory_written path={inventory_path} count={len(entries)}")


def build_prompt(prompt_path, source_md_abs_path, output_md_abs_path, inventory_json_abs_path):
    with open(prompt_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    replacements = {
        "[SOURCE_MD_ABS_PATH]": os.path.abspath(source_md_abs_path),
        "[OUTPUT_MD_ABS_PATH]": os.path.abspath(output_md_abs_path),
        "[INVENTORY_JSON_ABS_PATH]": os.path.abspath(inventory_json_abs_path),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def build_codex_exec_command(codex_cmd, codex_cwd, add_dirs, skip_git_check):
    args = shlex.split(codex_cmd)
    if not args:
        return []
    if "exec" not in args:
        args.extend(CODEX_EXEC_ARGS)
    if codex_cwd and "-C" not in args and "--cd" not in args:
        args.extend(["-C", codex_cwd])
    for add_dir in add_dirs:
        if add_dir:
            args.extend(["--add-dir", add_dir])
    if skip_git_check and "--skip-git-repo-check" not in args:
        args.append("--skip-git-repo-check")
    return args


def run_codex_exec(codex_cmd, prompt, codex_cwd, add_dirs, skip_git_check, log_path):
    args = build_codex_exec_command(codex_cmd, codex_cwd, add_dirs, skip_git_check)
    if not args:
        return False, "empty_codex_cmd"
    base_cmd = args[0]
    if shutil.which(base_cmd) is None:
        log_event(log_path, f"codex_missing cmd={base_cmd}")
        return False, "codex_missing"
    cmd_display = " ".join(shlex.quote(arg) for arg in args)
    log_event(log_path, f"codex_launch cmd={cmd_display}")
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
        log_event(log_path, f"codex_exception error={exc}")
        return False, str(exc)

    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=CODEX_EXEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - start_time
        log_event(log_path, f"codex_timeout seconds={elapsed:.1f}")
        return False, "timeout"
    except Exception as exc:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass
        log_event(log_path, f"codex_exception error={exc}")
        return False, str(exc)

    elapsed = time.monotonic() - start_time
    log_event(
        log_path,
        "codex_result "
        f"rc={proc.returncode} "
        f"seconds={elapsed:.1f} "
        f"stdout={sanitize_log_text(stdout, MAX_LOG_CHARS)} "
        f"stderr={sanitize_log_text(stderr, MAX_LOG_CHARS)}",
    )
    if proc.returncode != 0:
        return False, f"rc={proc.returncode}"
    return True, None


def make_temp_md_path(source_md_abs_path):
    folder = os.path.dirname(source_md_abs_path)
    base = os.path.basename(source_md_abs_path)
    while True:
        temp_name = f".{base}.ma-cleanup.tmp.{uuid.uuid4().hex}"
        temp_path = os.path.join(folder, temp_name)
        if not os.path.exists(temp_path):
            return temp_path


def remove_temp_md(path, log_path):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            log_event(log_path, f"temp_removed path={path}")
    except Exception as exc:
        log_event(log_path, f"temp_remove_failed path={path} error={exc}")


def cleanup_stale_temp_files(book_root, log_path):
    removed = 0
    for root, _, names in os.walk(book_root):
        for name in names:
            if ".ma-cleanup.tmp." not in name:
                continue
            path = os.path.join(root, name)
            try:
                os.remove(path)
                removed += 1
                log_event(log_path, f"temp_removed path={path}")
            except Exception as exc:
                log_event(log_path, f"temp_remove_failed path={path} error={exc}")
    if removed:
        log_event(log_path, f"temp_cleanup_removed count={removed}")


def resolve_sources(source_args, book_root, log_path):
    if not source_args:
        return sort_queue(list_markdown_files(book_root), SORT_MODE, log_path)
    resolved = []
    for source in source_args:
        candidate = source
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(os.path.join(book_root, source))
            if not os.path.exists(candidate):
                candidate = os.path.abspath(source)
        if os.path.isdir(candidate):
            resolved.extend(list_markdown_files(candidate))
        elif os.path.isfile(candidate) and candidate.lower().endswith(".md"):
            resolved.append(candidate)
        else:
            log_event(log_path, f"source_not_found path={source}")
    unique_paths = sorted({os.path.abspath(path) for path in resolved})
    return unique_paths


def main():
    parser = argparse.ArgumentParser(
        description="Clean Markdown OCR artifacts and links across The Math Academy Way.",
    )
    parser.add_argument("--book-root", default=BOOK_ROOT)
    parser.add_argument("--prompt-path", default=PROMPT_PATH)
    parser.add_argument("--inventory-path", default=INVENTORY_PATH)
    parser.add_argument("--state-path", default=STATE_PATH)
    parser.add_argument("--log-path", default=LOG_PATH)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--codex-cmd", default=CODEX_CLI_CMD)
    parser.add_argument("--sort-mode", default=SORT_MODE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-git-check", action="store_true")
    args = parser.parse_args()

    book_root = os.path.abspath(args.book_root)
    prompt_path = os.path.abspath(args.prompt_path)
    inventory_path = os.path.abspath(args.inventory_path)
    state_path = os.path.abspath(args.state_path)
    log_path = os.path.abspath(args.log_path)

    if not os.path.isdir(book_root):
        print(f"Book root not found: {book_root}")
        return 1
    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        return 1

    dry_run = DRY_RUN or args.dry_run
    overwrite = OVERWRITE or args.overwrite
    log_event(
        log_path,
        f"run_start book_root={book_root} dry_run={dry_run} overwrite={overwrite}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = find_git_root(script_dir) or find_git_root(book_root)
    if codex_cwd:
        log_event(log_path, f"codex_cwd path={codex_cwd}")
    else:
        log_event(log_path, "codex_cwd_missing")

    build_inventory(book_root, inventory_path, log_path)
    state = load_state(state_path, log_path)
    completed = state.get("completed", {})

    if CLEANUP_TEMP_ON_START and not dry_run:
        cleanup_stale_temp_files(book_root, log_path)

    sources = resolve_sources(args.source, book_root, log_path)
    sources = sort_queue(sources, args.sort_mode, log_path)
    if not sources:
        print("No Markdown files found.")
        log_event(log_path, "no_markdown_found")
        log_event(log_path, "run_end")
        return 0

    add_dirs = []
    if CODEX_ADD_DIR:
        add_dirs.extend([book_root, script_dir])
    skip_git_check = CODEX_SKIP_GIT_CHECK or args.skip_git_check

    total = len(sources)
    for index, source_path in enumerate(sources, start=1):
        source_path = os.path.abspath(source_path)
        rel_path = os.path.relpath(source_path, book_root)
        print(f"[{index}/{total}] {rel_path}")
        if not os.path.exists(source_path):
            log_event(log_path, f"source_missing path={source_path}")
            continue

        source_hash = sha256_file(source_path)
        state_entry = completed.get(source_path, {})
        if (
            not overwrite
            and isinstance(state_entry, dict)
            and state_entry.get("source_sha256") == source_hash
        ):
            log_event(log_path, f"skipped_state path={source_path}")
            continue

        temp_output = make_temp_md_path(source_path)
        prompt = build_prompt(prompt_path, source_path, temp_output, inventory_path)
        log_event(log_path, f"file_start path={source_path} temp={temp_output}")

        if dry_run:
            log_event(log_path, f"dry_run_prompt_ready path={source_path}")
            remove_temp_md(temp_output, log_path)
            continue

        success = False
        try:
            ok, error = run_codex_exec(
                args.codex_cmd,
                prompt,
                codex_cwd,
                add_dirs,
                skip_git_check,
                log_path,
            )
            if not ok:
                log_event(log_path, f"file_failed path={source_path} error={error}")
                continue
            if not os.path.exists(temp_output):
                log_event(log_path, f"temp_missing path={temp_output}")
                continue
            temp_size = os.path.getsize(temp_output)
            if temp_size < MIN_MD_BYTES:
                log_event(
                    log_path,
                    f"temp_too_small path={temp_output} bytes={temp_size}",
                )
                continue

            os.replace(temp_output, source_path)
            final_hash = sha256_file(source_path)
            completed[source_path] = {
                "completed_at": now_iso(),
                "source_sha256": final_hash,
            }
            state["completed"] = completed
            save_state(state_path, state, log_path)
            log_event(log_path, f"file_success path={source_path}")
            success = True
        finally:
            if not success:
                remove_temp_md(temp_output, log_path)

    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
