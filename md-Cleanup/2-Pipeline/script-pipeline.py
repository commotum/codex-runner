import argparse
import datetime
import os
import shutil
import subprocess
import sys


PREPROCESS_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "0-Preprocess", "script-preprocess.py")
PROCESS_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "1-Process", "script-process.py")
DEFAULT_BOOKS_IN = os.path.join(os.path.dirname(__file__), "..", "3-Books-In")
DEFAULT_BOOKS_OUT = os.path.join(os.path.dirname(__file__), "..", "4-Books-Out")
DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "pipeline.log")
STRUCTURED_DIRNAME = "_structured"
STRUCTURED_BUCKET = "Structured"
UNSTRUCTURED_BUCKET = "Un-Structured"


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path, message):
    line = f"{now_iso()} {message}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        sys.stderr.write(line)


def list_book_dirs(books_in):
    entries = []
    for name in sorted(os.listdir(books_in), key=str.lower):
        path = os.path.join(books_in, name)
        if os.path.isdir(path):
            entries.append(os.path.abspath(path))
    return entries


def run_command(cmd, log_path):
    log_event(log_path, f"command_start cmd={' '.join(cmd)}")
    result = subprocess.run(cmd)
    log_event(log_path, f"command_end rc={result.returncode}")
    return result.returncode


def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def move_book(book_dir, books_out, overwrite, log_path):
    book_name = os.path.basename(book_dir)
    structured_source = os.path.join(book_dir, STRUCTURED_DIRNAME)
    structured_bucket = os.path.join(books_out, STRUCTURED_BUCKET)
    unstructured_bucket = os.path.join(books_out, UNSTRUCTURED_BUCKET)
    structured_destination = os.path.join(structured_bucket, book_name)
    unstructured_destination = os.path.join(unstructured_bucket, book_name)

    os.makedirs(structured_bucket, exist_ok=True)
    os.makedirs(unstructured_bucket, exist_ok=True)

    for destination in [structured_destination, unstructured_destination]:
        if os.path.exists(destination):
            if not overwrite:
                raise FileExistsError(destination)
            remove_path(destination)

    if os.path.isdir(structured_source):
        shutil.move(structured_source, structured_destination)
        log_event(
            log_path,
            f"structured_moved source={structured_source} destination={structured_destination}",
        )

    os.makedirs(unstructured_destination, exist_ok=True)
    for name in os.listdir(book_dir):
        source_path = os.path.join(book_dir, name)
        destination_path = os.path.join(unstructured_destination, name)
        shutil.move(source_path, destination_path)

    try:
        os.rmdir(book_dir)
    except OSError:
        pass

    log_event(
        log_path,
        f"unstructured_moved source={book_dir} destination={unstructured_destination}",
    )


def main():
    parser = argparse.ArgumentParser(description="Orchestrate preprocess and cleanup for all incoming books.")
    parser.add_argument("--books-in", default=DEFAULT_BOOKS_IN)
    parser.add_argument("--books-out", default=DEFAULT_BOOKS_OUT)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--book", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    books_in = os.path.abspath(args.books_in)
    books_out = os.path.abspath(args.books_out)
    log_path = os.path.abspath(args.log_path)
    preprocess_script = os.path.abspath(PREPROCESS_SCRIPT)
    process_script = os.path.abspath(PROCESS_SCRIPT)

    if not os.path.isdir(books_in):
        print(f"Books-in directory not found: {books_in}")
        return 1
    os.makedirs(books_out, exist_ok=True)
    os.makedirs(os.path.join(books_out, STRUCTURED_BUCKET), exist_ok=True)
    os.makedirs(os.path.join(books_out, UNSTRUCTURED_BUCKET), exist_ok=True)

    candidates = list_book_dirs(books_in)
    if args.book:
        wanted = {os.path.abspath(os.path.join(books_in, item)) if not os.path.isabs(item) else os.path.abspath(item) for item in args.book}
        candidates = [path for path in candidates if path in wanted]

    if not candidates:
        print("No book directories found.")
        return 0

    log_event(log_path, f"run_start count={len(candidates)} dry_run={args.dry_run} overwrite={args.overwrite}")
    for index, book_dir in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {os.path.basename(book_dir)}")
        log_event(log_path, f"book_start path={book_dir}")

        preprocess_cmd = [sys.executable, preprocess_script, "--book-dir", book_dir]
        process_cmd = [sys.executable, process_script, "--book-dir", book_dir]
        if args.overwrite:
            process_cmd.append("--overwrite")
        if args.dry_run:
            preprocess_cmd.append("--dry-run")
            process_cmd.append("--dry-run")

        if run_command(preprocess_cmd, log_path) != 0:
            log_event(log_path, f"book_failed stage=preprocess path={book_dir}")
            return 1
        if run_command(process_cmd, log_path) != 0:
            log_event(log_path, f"book_failed stage=process path={book_dir}")
            return 1
        if args.dry_run:
            log_event(log_path, f"book_dry_run_complete path={book_dir}")
            continue

        try:
            move_book(book_dir, books_out, args.overwrite, log_path)
        except Exception as exc:
            print(f"Failed to move book: {book_dir} -> {exc}")
            log_event(log_path, f"book_failed stage=move path={book_dir} error={exc}")
            return 1

    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
