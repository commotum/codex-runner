# Fork Session

`fork_session.py` is a small CLI that takes an existing Codex session id,
launches `codex fork` in a detached `tmux` session, interrupts the forked
interactive session, and returns the new forked session id.

This exists because `codex fork` does not expose a documented structured output
format. The stable observable signal is the final line printed after interrupt:

```text
To continue this session, run codex resume <FORKED_SESSION_ID>
```

The script captures that line from the `tmux` pane transcript and parses the
forked id out of it. It is intentionally defensive:

- it uses a large detached `tmux` pane to reduce line wrapping
- it captures full pane scrollback, joining wrapped lines
- it polls for the resume line and returns immediately once it appears
- it interrupts based on transcript stall rather than relying on one long fixed wait
- it retries the whole fork operation if a transcript never yields a forked id
- it preserves a transcript path on final failure so callers have something to
  inspect

This was verified end to end against a fresh `codex exec` session and a
follow-up `codex exec resume` using the returned forked id.

## Requirements

- `codex` on `PATH`
- `tmux` on `PATH`
- access to the local Codex session store under `~/.codex`

In practice this usually means running outside the sandbox.

## Basic usage

Print only the forked session id:

```bash
python PIPELINE/Electrical-and-Computer-Engineering/Runners/fork_session.py <SESSION_ID>
```

Example:

```bash
python PIPELINE/Electrical-and-Computer-Engineering/Runners/fork_session.py 019d8365-8cf2-7f01-9fda-b693c702ad1a
```

The output is just the new forked id, which makes it convenient to capture in a
shell variable.

## JSON mode

If you want metadata instead of only the id:

```bash
python PIPELINE/Electrical-and-Computer-Engineering/Runners/fork_session.py \
  <SESSION_ID> \
  --json
```

Example output:

```json
{
  "attempt_count": 1,
  "attempts": [
    {
      "attempt_index": 1,
      "duration_seconds": 5.078,
      "error": "",
      "fork_banner_found": true,
      "resume_command_found": true,
      "tmux_session_name": "codexfork_02ad1a_ab12cd_a1"
    }
  ],
  "cwd": "/home/jake/Developer/MA",
  "fork_command": "codex fork --no-alt-screen 019d8365-8cf2-7f01-9fda-b693c702ad1a",
  "fork_banner_found": true,
  "forked_from_session_id": "019d8365-8cf2-7f01-9fda-b693c702ad1a",
  "forked_session_id": "019d8365-f279-70c1-99f1-a6a6e314c2fe",
  "resume_command_found": true,
  "source_session_id": "019d8365-8cf2-7f01-9fda-b693c702ad1a",
  "success": true,
  "tmux_session_name": "codexfork_02ad1a_ab12cd_a1"
}
```

## Failure output

In `--json` mode, the script still prints structured output if it fails, but it
exits with a nonzero status code.

The important failure flags are:

- `success`
- `fork_banner_found`
- `resume_command_found`
- `error`
- `attempt_count`
- `attempts`
- `transcript_path`

If the transcript does not contain the final `codex resume ...` line, you will
get a payload shaped like:

```json
{
  "attempt_count": 3,
  "attempts": [
    {
      "attempt_index": 1,
      "duration_seconds": 31.112,
      "error": "resume command was not found in the captured tmux transcript",
      "fork_banner_found": true,
      "resume_command_found": false,
      "tmux_session_name": "codexfork_02ad1a_ab12cd_a1"
    }
  ],
  "error": "failed to capture forked session id from tmux transcript; inspect /tmp/fork.attempt-03.txt",
  "fork_banner_found": true,
  "forked_from_session_id": "019d8365-8cf2-7f01-9fda-b693c702ad1a",
  "forked_session_id": "",
  "resume_command_found": false,
  "success": false,
  "transcript_path": "/tmp/fork.attempt-03.txt"
}
```

In plain mode, failures go to stderr and the script exits nonzero.

## Saving the transcript

If you want the captured pane transcript for debugging:

```bash
python PIPELINE/Electrical-and-Computer-Engineering/Runners/fork_session.py \
  <SESSION_ID> \
  --json \
  --transcript-path /tmp/codex-fork-transcript.txt
```

That transcript should include:

```text
Thread forked from <SESSION_ID>
To continue this session, run codex resume <FORKED_SESSION_ID>
```

## Useful options

- `--seconds-before-interrupt`
  Minimum startup grace before the helper begins interrupting a stalled forked
  session.
  Default: `1.0`

- `--seconds-after-interrupt`
  How long to wait after `Ctrl-C` before polling the pane again.
  Default: `0.25`

- `--max-wait-seconds`
  Maximum total time to wait for a resume command inside one fork attempt.
  Default: `12.0`

- `--poll-interval`
  How often to re-capture the pane while waiting for the resume command.
  Default: `0.2`

- `--retry-interrupt-interval`
  How long the transcript may stay unchanged before sending another `Ctrl-C`.
  Default: `0.75`

- `--fork-attempts`
  How many full fork attempts to make before failing.
  Default: `3`

- `--attempt-backoff-seconds`
  How long to wait between full fork attempts.
  Default: `1.0`

- `--cwd`
  Working directory for the detached `tmux` session. Default is the repo root.

- `--keep-tmux-session`
  Do not clean up the detached `tmux` session. Useful if you want to inspect it
  manually with `tmux attach -t <SESSION_NAME>`.

## End-to-end pattern

This script is useful in a three-step flow:

1. Start a headless session with `codex exec`.
2. Fork that session with `fork_session.py`.
3. Continue from the forked id with `codex exec resume`.

Example:

```bash
FORKED_ID="$(python PIPELINE/Electrical-and-Computer-Engineering/Runners/fork_session.py <SESSION_ID>)"

codex exec resume --json --full-auto "$FORKED_ID" -
```

## Caveat

This is a pragmatic CLI wrapper, not a documented Codex JSON API for forking.
It depends on the current `codex fork` terminal text continuing to include the
`codex resume <FORKED_SESSION_ID>` line after interruption.

## Fault tolerance

The current helper is meant to fail in a way that downstream runners can reason
about:

- transient transcript misses are retried automatically
- line-wrapping issues are reduced by using a larger pane and joined capture
- JSON mode keeps attempt metadata so callers can distinguish "never forked" vs
  "forked but never exposed a resume id"
- final failures preserve a transcript path for manual inspection

This still does not guarantee forward progress by itself. A runner that uses
this helper still needs its own retry, skip, and continue-on-error policy if it
should move on to the next document after a failed fork.
