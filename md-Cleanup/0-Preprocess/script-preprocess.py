import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import time


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt-preprocess.md")
CODEX_CLI_CMD = "codex"
CODEX_EXEC_ARGS = ["exec", "--full-auto"]
CODEX_EXEC_TIMEOUT = 7200
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
MAX_LOG_CHARS = 4000
DEFAULT_STRUCTURED_DIRNAME = "_structured"
DEFAULT_PROGRESS_FILENAME = "book_progress.json"
DEFAULT_LOG_FILENAME = "preprocess.log"
DEFAULT_MANIFEST_FILENAME = "preprocess_manifest.json"


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


def find_source_markdown(book_dir):
    candidates = []
    for name in os.listdir(book_dir):
        path = os.path.join(book_dir, name)
        if not os.path.isfile(path) or not name.lower().endswith(".md"):
            continue
        candidates.append(path)
    if not candidates:
        return None

    base_name = os.path.basename(book_dir).strip().lower()
    for path in candidates:
        stem = os.path.splitext(os.path.basename(path))[0].strip().lower()
        if stem == base_name:
            return os.path.abspath(path)

    candidates.sort(key=lambda path: (os.path.getsize(path), path), reverse=True)
    return os.path.abspath(candidates[0])


def build_prompt(prompt_path, source_md_abs_path, structured_root_abs_path, progress_json_abs_path):
    with open(prompt_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    replacements = {
        "[SOURCE_MD_ABS_PATH]": os.path.abspath(source_md_abs_path),
        "[STRUCTURED_ROOT_ABS_PATH]": os.path.abspath(structured_root_abs_path),
        "[PROGRESS_JSON_ABS_PATH]": os.path.abspath(progress_json_abs_path),
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


def write_manifest(manifest_path, payload):
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess one OCR-extracted book into section files.")
    parser.add_argument("--book-dir", required=True)
    parser.add_argument("--prompt-path", default=PROMPT_PATH)
    parser.add_argument("--structured-dirname", default=DEFAULT_STRUCTURED_DIRNAME)
    parser.add_argument("--progress-filename", default=DEFAULT_PROGRESS_FILENAME)
    parser.add_argument("--manifest-filename", default=DEFAULT_MANIFEST_FILENAME)
    parser.add_argument("--log-filename", default=DEFAULT_LOG_FILENAME)
    parser.add_argument("--codex-cmd", default=CODEX_CLI_CMD)
    parser.add_argument("--skip-git-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    book_dir = os.path.abspath(args.book_dir)
    prompt_path = os.path.abspath(args.prompt_path)
    if not os.path.isdir(book_dir):
        print(f"Book directory not found: {book_dir}")
        return 1
    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        return 1

    source_md_path = find_source_markdown(book_dir)
    if source_md_path is None:
        print(f"No source Markdown found in: {book_dir}")
        return 1

    structured_root = os.path.join(book_dir, args.structured_dirname)
    progress_json_path = os.path.join(book_dir, args.progress_filename)
    log_path = os.path.join(book_dir, args.log_filename)
    manifest_path = os.path.join(book_dir, args.manifest_filename)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = find_git_root(script_dir) or find_git_root(book_dir)

    manifest = {
        "book_dir": book_dir,
        "source_markdown": source_md_path,
        "structured_root": structured_root,
        "progress_json": progress_json_path,
        "prompt_path": prompt_path,
        "generated_at": now_iso(),
    }
    write_manifest(manifest_path, manifest)
    log_event(log_path, f"run_start book_dir={book_dir} dry_run={args.dry_run}")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        log_event(log_path, "dry_run_end")
        return 0

    prompt = build_prompt(prompt_path, source_md_path, structured_root, progress_json_path)
    add_dirs = [book_dir, script_dir] if CODEX_ADD_DIR else []
    ok, error = run_codex_exec(
        args.codex_cmd,
        prompt,
        codex_cwd,
        add_dirs,
        CODEX_SKIP_GIT_CHECK or args.skip_git_check,
        log_path,
    )
    if not ok:
        print(f"Preprocess failed: {error}")
        log_event(log_path, f"run_failed error={error}")
        return 1

    if not os.path.isdir(structured_root):
        print(f"Structured root was not created: {structured_root}")
        log_event(log_path, "run_failed missing_structured_root")
        return 1
    if not os.path.isfile(progress_json_path):
        print(f"Progress JSON was not created: {progress_json_path}")
        log_event(log_path, "run_failed missing_progress_json")
        return 1

    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
