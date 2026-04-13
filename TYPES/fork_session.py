#!/usr/bin/env python3

import argparse
import json
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


RUNNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = RUNNER_DIR / ".runtime" / "fork-session"
FORKED_FROM_RE = re.compile(
    r"Thread forked from\s+([0-9a-f\-\s]{36,64})", re.IGNORECASE
)
RESUME_RE = re.compile(
    r"To continue this session,\s*run codex resume\s+([0-9a-f\-\s]{36,64})",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fork a Codex session, interrupt the forked interactive session, "
            "and print the new forked session id."
        )
    )
    parser.add_argument("session_id", help="Existing Codex session id to fork.")
    parser.add_argument(
        "--seconds-before-interrupt",
        type=float,
        default=1.0,
        help="Minimum startup grace before the helper begins interrupting a stalled forked session.",
    )
    parser.add_argument(
        "--seconds-after-interrupt",
        type=float,
        default=0.25,
        help="How long to wait after Ctrl-C before polling the pane again.",
    )
    parser.add_argument(
        "--cwd",
        default=str(REPO_ROOT),
        help="Working directory for the detached tmux session.",
    )
    parser.add_argument(
        "--session-name",
        help="Optional tmux session name. Defaults to a generated unique name.",
    )
    parser.add_argument(
        "--transcript-path",
        help="Optional file path to save the captured tmux pane transcript.",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=12.0,
        help="Maximum total time to wait for the fork transcript to reveal a resume command.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="How often to re-capture the tmux pane while waiting for the resume command.",
    )
    parser.add_argument(
        "--retry-interrupt-interval",
        type=float,
        default=0.75,
        help="How long the transcript may stay unchanged before sending another Ctrl-C.",
    )
    parser.add_argument(
        "--fork-attempts",
        type=int,
        default=3,
        help="How many full fork attempts to make before giving up.",
    )
    parser.add_argument(
        "--attempt-backoff-seconds",
        type=float,
        default=1.0,
        help="How long to wait between full fork attempts after a failed capture.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON object instead of only the forked session id.",
    )
    parser.add_argument(
        "--keep-tmux-session",
        action="store_true",
        help="Keep the detached tmux session after capture for manual inspection.",
    )
    return parser.parse_args()


def ensure_dependency(name):
    if shutil.which(name):
        return
    raise SystemExit(f"required command not found on PATH: {name}")


def make_tmux_session_name(source_session_id):
    return f"codexfork_{source_session_id.replace('-', '')[-8:]}_{secrets.token_hex(3)}"


def run_tmux(*args, check=True):
    result = subprocess.run(
        ["tmux", *args],
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if check and result.returncode != 0:
        joined = " ".join(shlex.quote(part) for part in ("tmux", *args))
        raise RuntimeError(
            f"{joined} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def save_transcript(path, text):
    transcript_path = Path(path).resolve()
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")
    return transcript_path


def normalize_session_id(value):
    compact = re.sub(r"\s+", "", value or "")
    return compact if re.fullmatch(r"[0-9a-f-]{36}", compact) else ""


def parse_fork_output(text):
    normalized = text.replace("\r", "")
    forked_from_match = FORKED_FROM_RE.search(normalized)
    resume_match = RESUME_RE.search(normalized)
    forked_from = normalize_session_id(forked_from_match.group(1)) if forked_from_match else ""
    forked_session_id = normalize_session_id(resume_match.group(1)) if resume_match else ""
    return forked_from, forked_session_id


def capture_tmux_transcript(tmux_session_name):
    return run_tmux(
        "capture-pane", "-J", "-p", "-S", "-32768", "-t", tmux_session_name
    ).stdout


def wait_for_resume_command(tmux_session_name, args):
    transcript_text = ""
    previous_transcript_text = ""
    forked_from_session_id = ""
    forked_session_id = ""
    started_at = time.monotonic()
    last_change_at = started_at
    last_interrupt_at = None
    deadline = time.monotonic() + max(
        args.max_wait_seconds,
        args.seconds_before_interrupt + args.seconds_after_interrupt,
    )
    startup_grace = max(args.seconds_before_interrupt, 0.0)
    stall_interval = max(args.retry_interrupt_interval, 0.1)
    poll_interval = max(args.poll_interval, 0.1)

    while True:
        now = time.monotonic()
        transcript_text = capture_tmux_transcript(tmux_session_name)
        if transcript_text != previous_transcript_text:
            previous_transcript_text = transcript_text
            last_change_at = now
        forked_from_session_id, forked_session_id = parse_fork_output(transcript_text)
        if forked_session_id:
            return transcript_text, forked_from_session_id, forked_session_id

        if now >= deadline:
            return transcript_text, forked_from_session_id, forked_session_id

        transcript_is_stalled = (now - last_change_at) >= stall_interval
        past_startup_grace = (now - started_at) >= startup_grace
        can_interrupt_again = last_interrupt_at is None or (now - last_interrupt_at) >= stall_interval
        if past_startup_grace and transcript_is_stalled and can_interrupt_again:
            run_tmux("send-keys", "-t", tmux_session_name, "C-c")
            last_interrupt_at = time.monotonic()
            if args.seconds_after_interrupt > 0:
                time.sleep(args.seconds_after_interrupt)
            continue

        sleep_for = min(poll_interval, max(0.0, deadline - now))
        if sleep_for > 0:
            time.sleep(sleep_for)


def format_exception_message(exc):
    message = str(exc).strip() or exc.__class__.__name__
    if "error connecting to /tmp/tmux-" in message and "Operation not permitted" in message:
        return (
            "tmux could not access its server socket; this usually means the command "
            "is running in a restricted sandbox and should be retried outside it"
        )
    return message


def explicit_attempt_transcript_path(path, attempt_index, attempts):
    transcript_path = Path(path).resolve()
    if attempts <= 1:
        return transcript_path
    suffix = "".join(transcript_path.suffixes)
    stem = transcript_path.name[: -len(suffix)] if suffix else transcript_path.name
    filename = f"{stem}.attempt-{attempt_index:02d}{suffix}"
    return transcript_path.with_name(filename)


def auto_failure_transcript_path(source_session_id, attempt_index):
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    slug = source_session_id.replace("-", "")[-12:]
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR / f"{slug}-attempt-{attempt_index:02d}-{stamp}.txt"


def run_single_fork_attempt(source_session_id, cwd, tmux_session_name, fork_command, args):
    transcript_text = ""
    forked_from_session_id = ""
    forked_session_id = ""
    try:
        run_tmux("kill-session", "-t", tmux_session_name, check=False)
        run_tmux(
            "new-session",
            "-d",
            "-s",
            tmux_session_name,
            "-x",
            "240",
            "-y",
            "80",
            "-c",
            str(cwd),
        )
        run_tmux("set-option", "-t", tmux_session_name, "remain-on-exit", "on")
        run_tmux("send-keys", "-t", tmux_session_name, fork_command, "C-m")
        (
            transcript_text,
            forked_from_session_id,
            forked_session_id,
        ) = wait_for_resume_command(tmux_session_name, args)
    finally:
        if not args.keep_tmux_session:
            run_tmux("kill-session", "-t", tmux_session_name, check=False)
    return transcript_text, forked_from_session_id, forked_session_id


def build_result_payload(
    *,
    source_session_id,
    forked_from_session_id,
    forked_session_id,
    tmux_session_name,
    fork_command,
    cwd,
    transcript_path,
    transcript_text,
    error,
    attempts,
):
    payload = {
        "success": not error,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "source_session_id": source_session_id,
        "forked_from_session_id": forked_from_session_id,
        "forked_session_id": forked_session_id,
        "fork_command": fork_command,
        "cwd": str(cwd),
        "tmux_session_name": tmux_session_name,
        "fork_banner_found": bool(forked_from_session_id),
        "resume_command_found": bool(forked_session_id),
    }
    if transcript_path:
        payload["transcript_path"] = str(transcript_path)
    else:
        payload["transcript_preview"] = "\n".join(transcript_text.splitlines()[-12:])
    if error:
        payload["error"] = error
    return payload


def main():
    args = parse_args()
    ensure_dependency("codex")
    ensure_dependency("tmux")

    source_session_id = args.session_id
    cwd = Path(args.cwd).resolve()
    base_tmux_session_name = args.session_name or make_tmux_session_name(source_session_id)
    tmux_session_name = base_tmux_session_name
    fork_command = f"codex fork --no-alt-screen {shlex.quote(source_session_id)}"

    transcript_text = ""
    forked_from_session_id = ""
    forked_session_id = ""
    attempts = max(args.fork_attempts, 1)
    attempt_records = []
    for attempt_index in range(1, attempts + 1):
        attempt_session_name = base_tmux_session_name
        if attempts > 1:
            attempt_session_name = f"{base_tmux_session_name}_a{attempt_index}"
        started_at = time.monotonic()
        attempt_error = ""
        attempt_transcript_path = None
        try:
            (
                transcript_text,
                forked_from_session_id,
                forked_session_id,
            ) = run_single_fork_attempt(
                source_session_id=source_session_id,
                cwd=cwd,
                tmux_session_name=attempt_session_name,
                fork_command=fork_command,
                args=args,
            )
            if not forked_session_id:
                attempt_error = "resume command was not found in the captured tmux transcript"
        except Exception as exc:
            transcript_text = ""
            forked_from_session_id = ""
            forked_session_id = ""
            attempt_error = format_exception_message(exc)

        if args.transcript_path:
            attempt_transcript_path = save_transcript(
                explicit_attempt_transcript_path(args.transcript_path, attempt_index, attempts),
                transcript_text,
            )

        attempt_records.append(
            {
                "attempt_index": attempt_index,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "error": attempt_error,
                "fork_banner_found": bool(forked_from_session_id),
                "resume_command_found": bool(forked_session_id),
                "tmux_session_name": attempt_session_name,
                **(
                    {"transcript_path": str(attempt_transcript_path)}
                    if attempt_transcript_path is not None
                    else {}
                ),
            }
        )
        tmux_session_name = attempt_session_name
        if forked_session_id:
            break
        if attempt_index < attempts and args.attempt_backoff_seconds > 0:
            time.sleep(args.attempt_backoff_seconds)

    transcript_path = None
    if attempt_records:
        last_attempt = attempt_records[-1]
        if "transcript_path" in last_attempt:
            transcript_path = Path(last_attempt["transcript_path"])
    if transcript_path is None and not forked_session_id and transcript_text:
        transcript_path = save_transcript(
            auto_failure_transcript_path(source_session_id, len(attempt_records) or 1),
            transcript_text,
        )

    error = ""
    last_attempt_error = ""
    for attempt in reversed(attempt_records):
        if attempt.get("error"):
            last_attempt_error = attempt["error"]
            break
    if not forked_session_id:
        if transcript_path and transcript_text:
            error = (
                "failed to capture forked session id from tmux transcript; "
                f"inspect {transcript_path}"
            )
        elif last_attempt_error:
            error = last_attempt_error
        else:
            error = (
                "failed to capture forked session id from tmux transcript; "
                "rerun with --transcript-path to inspect the transcript"
            )

    payload = build_result_payload(
        source_session_id=source_session_id,
        forked_from_session_id=forked_from_session_id,
        forked_session_id=forked_session_id,
        tmux_session_name=tmux_session_name,
        fork_command=fork_command,
        cwd=cwd,
        transcript_path=transcript_path,
        transcript_text=transcript_text,
        error=error,
        attempts=attempt_records,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        if error:
            sys.exit(1)
        return

    if error:
        print(error, file=sys.stderr)
        sys.exit(1)

    print(forked_session_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
