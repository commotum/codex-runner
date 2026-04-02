import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt-process.md")
CODEX_CLI_CMD = "codex"
CODEX_EXEC_ARGS = ["exec", "--full-auto"]
CODEX_EXEC_TIMEOUT = 7200
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
MAX_LOG_CHARS = 4000
MIN_MD_BYTES = 20
DEFAULT_STRUCTURED_DIRNAME = "_structured"
DEFAULT_PROGRESS_FILENAME = "book_progress.json"
DEFAULT_INVENTORY_FILENAME = "process_inventory.json"
DEFAULT_LOG_FILENAME = "process.log"


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path, message):
    line = f"{now_iso()} {message}\n"
    try:
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_inventory(structured_root, inventory_path):
    documents = []
    for root, _, names in os.walk(structured_root):
        for name in names:
            if not name.lower().endswith(".md"):
                continue
            path = os.path.join(root, name)
            documents.append(
                {
                    "absolute_path": os.path.abspath(path),
                    "relative_path": os.path.relpath(path, structured_root).replace(os.sep, "/"),
                    "directory": os.path.relpath(os.path.dirname(path), structured_root).replace(os.sep, "/"),
                    "file_name": name,
                }
            )
    payload = {
        "generated_at": now_iso(),
        "structured_root": os.path.abspath(structured_root),
        "documents": sorted(documents, key=lambda item: item["relative_path"].lower()),
    }
    save_json(inventory_path, payload)


def build_prompt(prompt_path, source_md_abs_path, output_md_abs_path, progress_json_abs_path, inventory_json_abs_path):
    with open(prompt_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    replacements = {
        "[SOURCE_MD_ABS_PATH]": os.path.abspath(source_md_abs_path),
        "[OUTPUT_MD_ABS_PATH]": os.path.abspath(output_md_abs_path),
        "[PROGRESS_JSON_ABS_PATH]": os.path.abspath(progress_json_abs_path),
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
    log_event(log_path, f"codex_launch cmd={' '.join(shlex.quote(arg) for arg in args)}")
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
        temp_name = f".{base}.process.tmp.{uuid.uuid4().hex}"
        temp_path = os.path.join(folder, temp_name)
        if not os.path.exists(temp_path):
            return temp_path


def remove_temp(path):
    if path and os.path.exists(path):
        os.remove(path)


def iter_sections(progress_payload, structured_root):
    sections = progress_payload.get("sections", [])
    for section in sections:
        markdown_path = section.get("markdown_path")
        if not markdown_path:
            continue
        if os.path.isabs(markdown_path):
            absolute_path = markdown_path
        else:
            absolute_path = os.path.join(structured_root, markdown_path)
        yield section, os.path.abspath(absolute_path)


def main():
    parser = argparse.ArgumentParser(description="Clean extracted section Markdown for one book.")
    parser.add_argument("--book-dir", required=True)
    parser.add_argument("--prompt-path", default=PROMPT_PATH)
    parser.add_argument("--structured-dirname", default=DEFAULT_STRUCTURED_DIRNAME)
    parser.add_argument("--progress-filename", default=DEFAULT_PROGRESS_FILENAME)
    parser.add_argument("--inventory-filename", default=DEFAULT_INVENTORY_FILENAME)
    parser.add_argument("--log-filename", default=DEFAULT_LOG_FILENAME)
    parser.add_argument("--codex-cmd", default=CODEX_CLI_CMD)
    parser.add_argument("--skip-git-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    book_dir = os.path.abspath(args.book_dir)
    prompt_path = os.path.abspath(args.prompt_path)
    structured_root = os.path.join(book_dir, args.structured_dirname)
    progress_json_path = os.path.join(book_dir, args.progress_filename)
    inventory_json_path = os.path.join(book_dir, args.inventory_filename)
    log_path = os.path.join(book_dir, args.log_filename)

    if not os.path.isdir(book_dir):
        print(f"Book directory not found: {book_dir}")
        return 1
    if not os.path.isdir(structured_root):
        print(f"Structured root not found: {structured_root}")
        return 1
    if not os.path.isfile(progress_json_path):
        print(f"Progress JSON not found: {progress_json_path}")
        return 1
    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        return 1

    progress_payload = load_json(progress_json_path)
    build_inventory(structured_root, inventory_json_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = find_git_root(script_dir) or find_git_root(book_dir)

    add_dirs = [book_dir, structured_root, script_dir] if CODEX_ADD_DIR else []
    skip_git_check = CODEX_SKIP_GIT_CHECK or args.skip_git_check

    sections = list(iter_sections(progress_payload, structured_root))
    total = len(sections)
    if total == 0:
        print("No sections found in progress JSON.")
        return 1

    log_event(log_path, f"run_start book_dir={book_dir} dry_run={args.dry_run} overwrite={args.overwrite}")

    changed = False
    for index, (section, source_path) in enumerate(sections, start=1):
        if not os.path.isfile(source_path):
            log_event(log_path, f"section_missing path={source_path}")
            continue
        print(f"[{index}/{total}] {os.path.relpath(source_path, structured_root)}")
        source_hash = sha256_file(source_path)
        if not args.overwrite and section.get("cleaned") == source_hash:
            log_event(log_path, f"skipped_cleaned path={source_path}")
            continue

        temp_output = make_temp_md_path(source_path)
        prompt = build_prompt(prompt_path, source_path, temp_output, progress_json_path, inventory_json_path)
        if args.dry_run:
            log_event(log_path, f"dry_run_prompt_ready path={source_path}")
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
                log_event(log_path, f"section_failed path={source_path} error={error}")
                continue
            if not os.path.exists(temp_output):
                log_event(log_path, f"temp_missing path={temp_output}")
                continue
            if os.path.getsize(temp_output) < MIN_MD_BYTES:
                log_event(log_path, f"temp_too_small path={temp_output}")
                continue

            os.replace(temp_output, source_path)
            section["cleaned"] = sha256_file(source_path)
            section["cleaned_at"] = now_iso()
            changed = True
            success = True
            log_event(log_path, f"section_success path={source_path}")
        finally:
            if not success:
                try:
                    remove_temp(temp_output)
                except Exception as exc:
                    log_event(log_path, f"temp_remove_failed path={temp_output} error={exc}")

    if changed and not args.dry_run:
        save_json(progress_json_path, progress_payload)
    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
