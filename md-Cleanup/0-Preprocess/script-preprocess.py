import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from html import unescape

try:
    from tqdm import tqdm
except Exception:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "Progress"
            self.unit = unit or "item"
            self.count = 0
            if self.total is not None:
                print(f"{self.desc}: 0/{self.total} {self.unit}")

        def update(self, n=1):
            self.count += n
            if self.total is not None:
                print(f"{self.desc}: {self.count}/{self.total} {self.unit}")

        def close(self):
            return None

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.count += 1
                if self.total is not None:
                    print(f"{self.desc}: {self.count}/{self.total} {self.unit}")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(SCRIPT_DIR, "prompt-preprocess.md")
SCHEMA_PATH = os.path.join(SCRIPT_DIR, "toc-schema.json")
CODEX_CLI_CMD = "codex"
CODEX_EXEC_ARGS = ["exec"]
CODEX_EXEC_TIMEOUT = 3600
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
MAX_LOG_CHARS = 4000
DEFAULT_STRUCTURED_DIRNAME = "_structured"
DEFAULT_PROGRESS_FILENAME = "book_progress.json"
DEFAULT_LOG_FILENAME = "preprocess.log"
DEFAULT_MANIFEST_FILENAME = "preprocess_manifest.json"
DEFAULT_INVENTORY_FILENAME = "preprocess_inventory.json"
MIN_SECTION_BYTES = 20

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")
CHAPTER_PATTERN = re.compile(r"chapter\s+(\d+)\b", re.IGNORECASE)


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


def load_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.readlines()


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def normalize_text(value):
    value = unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`~]", "", value)
    value = value.replace("&", " and ")
    value = value.replace("—", "-").replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    value = value.strip().strip("\"'[]()")
    return value


def canonical_key(value):
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value):
    value = normalize_text(value)
    value = value.replace("&", "and")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "Section"


def natural_section_filename(folder_slug, title):
    normalized = canonical_key(title)
    if normalized in {"contents", "table of contents"}:
        return "Contents.md"
    if normalized == "preface":
        return "Preface.md"
    if normalized == "prologue":
        return "Prologue.md"
    if normalized == "epilogue":
        return "Epilogue.md"
    if normalized == "index":
        return "Index.md"
    return f"{folder_slug}.md"


def build_prompt(prompt_path, source_md_abs_path):
    template = load_text(prompt_path)
    return template.replace("[SOURCE_MD_ABS_PATH]", os.path.abspath(source_md_abs_path))


def build_codex_exec_command(codex_cmd, codex_cwd, add_dirs, skip_git_check, schema_path):
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
    args.extend(["--output-schema", schema_path])
    if skip_git_check and "--skip-git-repo-check" not in args:
        args.append("--skip-git-repo-check")
    return args


def run_codex_exec_json(codex_cmd, prompt, codex_cwd, add_dirs, skip_git_check, schema_path, log_path):
    args = build_codex_exec_command(codex_cmd, codex_cwd, add_dirs, skip_git_check, schema_path)
    if not args:
        return None, "empty_codex_cmd"
    base_cmd = args[0]
    if shutil.which(base_cmd) is None:
        log_event(log_path, f"codex_missing cmd={base_cmd}")
        return None, "codex_missing"
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
        return None, str(exc)

    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=CODEX_EXEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - start_time
        log_event(log_path, f"codex_timeout seconds={elapsed:.1f}")
        return None, "timeout"
    except Exception as exc:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass
        log_event(log_path, f"codex_exception error={exc}")
        return None, str(exc)

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
        return None, f"rc={proc.returncode}"
    try:
        return json.loads(stdout), None
    except Exception as exc:
        log_event(log_path, f"codex_json_parse_failed error={exc}")
        return None, "invalid_json"


def write_manifest(manifest_path, payload):
    save_json(manifest_path, payload)


def load_progress(progress_json_path, source_md_path, structured_root):
    if os.path.exists(progress_json_path):
        with open(progress_json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "book_title": os.path.splitext(os.path.basename(source_md_path))[0],
        "source_markdown": os.path.abspath(source_md_path),
        "structured_root": os.path.abspath(structured_root),
        "generated_at": now_iso(),
        "updated_at": now_iso(),
        "contents_heading": None,
        "stages": {
            "structured_root_created": False,
            "contents_extracted": False,
            "section_folders_created": False,
            "sections_extracted": False,
        },
        "sections": [],
    }


def update_progress(progress_json_path, progress):
    progress["updated_at"] = now_iso()
    save_json(progress_json_path, progress)


def parse_headings(lines):
    headings = []
    for index, line in enumerate(lines):
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        title = normalize_text(match.group(1))
        headings.append(
            {
                "line_index": index,
                "line_number": index + 1,
                "raw": line.rstrip("\n"),
                "title": title,
                "key": canonical_key(title),
            }
        )
    return headings


def find_contents_heading(lines):
    for index, line in enumerate(lines):
        text = canonical_key(line)
        if text in {"contents", "table of contents"}:
            return index
    return None


def infer_group_slug(group_name):
    group_slug = slugify(group_name).upper()
    return group_slug or "CHAPTERS"


def build_sections_from_toc(toc_payload):
    sections = []
    for index, item in enumerate(toc_payload.get("sections", []), start=1):
        title = normalize_text(item.get("title", ""))
        group = infer_group_slug(item.get("group", "CHAPTERS"))
        folder_slug = item.get("folder_slug") or slugify(title)
        folder_slug = slugify(folder_slug)
        file_name = natural_section_filename(folder_slug, title)
        sections.append(
            {
                "id": f"section-{index:03d}",
                "title": title,
                "group": group,
                "folder": folder_slug,
                "markdown_path": os.path.join(group, folder_slug, file_name).replace(os.sep, "/"),
                "source_heading": title,
                "kind": item.get("kind", "section"),
                "extracted": False,
                "cleaned": None,
                "start_line": None,
                "end_line": None,
                "image_count": 0,
            }
        )
    return sections


def extract_contents_to_progress(lines, headings, progress, structured_root, log_path):
    contents_heading_index = find_contents_heading(lines)
    contents_dir = os.path.join(structured_root, "FRONT-MATTER", "Contents")
    os.makedirs(contents_dir, exist_ok=True)
    contents_md_path = os.path.join(contents_dir, "Contents.md")

    contents_lines = []
    if contents_heading_index is not None:
        next_heading_index = None
        for heading in headings:
            if heading["line_index"] <= contents_heading_index:
                continue
            next_heading_index = heading["line_index"]
            break
        stop_index = next_heading_index if next_heading_index is not None else len(lines)
        contents_lines = lines[contents_heading_index:stop_index]
        progress["contents_heading"] = normalize_text(lines[contents_heading_index])
    else:
        progress["contents_heading"] = None
        log_event(log_path, "contents_heading_missing")

    temp_path = contents_md_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.writelines(contents_lines or ["# Contents\n"])
    os.replace(temp_path, contents_md_path)
    return contents_md_path


def ensure_section_folders(progress, structured_root):
    for section in progress.get("sections", []):
        markdown_path = os.path.join(structured_root, section["markdown_path"])
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)


def section_candidates(section):
    title = section["source_heading"]
    key = canonical_key(title)
    candidates = {key}

    chapter_match = CHAPTER_PATTERN.search(title)
    if chapter_match:
        number = chapter_match.group(1)
        candidates.add(canonical_key(f"chapter {number}"))
    simple = canonical_key(re.sub(r"^chapter\s+\d+\s*[-.:]?\s*", "", title, flags=re.IGNORECASE))
    if simple:
        candidates.add(simple)
    return [item for item in candidates if item]


def find_section_start_map(headings, sections):
    taken = set()
    start_map = {}
    for section in sections:
        candidates = section_candidates(section)
        best_heading = None
        for heading in headings:
            if heading["line_index"] in taken:
                continue
            heading_key = heading["key"]
            for candidate in candidates:
                if heading_key == candidate or heading_key.startswith(candidate) or candidate.startswith(heading_key):
                    best_heading = heading
                    break
            if best_heading is not None:
                break
        if best_heading is not None:
            start_map[section["id"]] = best_heading
            taken.add(best_heading["line_index"])
    return start_map


def find_local_image_paths(markdown_text, book_dir):
    discovered = []
    for raw_path in IMAGE_PATTERN.findall(markdown_text):
        path = raw_path.strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
        if not path or "://" in path or path.startswith("/"):
            continue
        abs_path = os.path.abspath(os.path.join(book_dir, path))
        if os.path.isfile(abs_path):
            discovered.append((raw_path, abs_path))
    return discovered


def copy_images_and_rewrite(markdown_text, source_path, book_dir):
    image_refs = find_local_image_paths(markdown_text, book_dir)
    if not image_refs:
        return markdown_text, 0

    images_dir = os.path.join(os.path.dirname(source_path), "Images")
    os.makedirs(images_dir, exist_ok=True)
    rewrites = {}
    for raw_path, abs_path in image_refs:
        destination = os.path.join(images_dir, os.path.basename(abs_path))
        if not os.path.exists(destination):
            shutil.copy2(abs_path, destination)
        rewrites[raw_path] = f"Images/{os.path.basename(abs_path)}"

    for old, new in rewrites.items():
        markdown_text = markdown_text.replace(f"({old})", f"({new})")
    return markdown_text, len(rewrites)


def write_atomic(path, text):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temp_path, path)


def build_contents_navigation(progress, structured_root):
    groups = {}
    for section in progress.get("sections", []):
        groups.setdefault(section["group"], []).append(section)

    contents_md_path = os.path.join(structured_root, "FRONT-MATTER", "Contents", "Contents.md")
    lines = ["# Contents", ""]
    for group_name in groups:
        lines.append(f"## {group_name}")
        lines.append("")
        for section in groups[group_name]:
            target = os.path.join(structured_root, section["markdown_path"])
            relative_target = os.path.relpath(target, os.path.dirname(contents_md_path)).replace(os.sep, "/")
            lines.append(f"- [{section['title']}]({relative_target})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def extract_sections(progress, lines, headings, structured_root, book_dir, log_path):
    sections = progress.get("sections", [])
    start_map = find_section_start_map(headings, sections)

    ordered = []
    for section in sections:
        heading = start_map.get(section["id"])
        if heading is None:
            continue
        ordered.append((section, heading))
    ordered.sort(key=lambda item: item[1]["line_index"])

    for index, (section, heading) in enumerate(tqdm(ordered, desc="Extract Sections", unit="section")):
        if section.get("extracted"):
            continue
        start_index = heading["line_index"]
        next_start = len(lines)
        if index + 1 < len(ordered):
            next_start = ordered[index + 1][1]["line_index"]
        body_lines = lines[start_index:next_start]
        if not body_lines:
            log_event(log_path, f"section_empty title={section['title']}")
            continue

        markdown_text = "".join(body_lines)
        target_path = os.path.join(structured_root, section["markdown_path"])
        markdown_text, image_count = copy_images_and_rewrite(markdown_text, target_path, book_dir)
        if len(markdown_text.encode("utf-8")) < MIN_SECTION_BYTES:
            log_event(log_path, f"section_too_small title={section['title']}")
            continue

        write_atomic(target_path, markdown_text)
        section["extracted"] = True
        section["start_line"] = start_index + 1
        section["end_line"] = next_start
        section["image_count"] = image_count


def build_preprocess_inventory(progress, inventory_path):
    payload = {
        "generated_at": now_iso(),
        "sections": progress.get("sections", []),
    }
    save_json(inventory_path, payload)


def main():
    parser = argparse.ArgumentParser(description="Preprocess one OCR-extracted book into section files.")
    parser.add_argument("--book-dir", required=True)
    parser.add_argument("--prompt-path", default=PROMPT_PATH)
    parser.add_argument("--schema-path", default=SCHEMA_PATH)
    parser.add_argument("--structured-dirname", default=DEFAULT_STRUCTURED_DIRNAME)
    parser.add_argument("--progress-filename", default=DEFAULT_PROGRESS_FILENAME)
    parser.add_argument("--manifest-filename", default=DEFAULT_MANIFEST_FILENAME)
    parser.add_argument("--inventory-filename", default=DEFAULT_INVENTORY_FILENAME)
    parser.add_argument("--log-filename", default=DEFAULT_LOG_FILENAME)
    parser.add_argument("--codex-cmd", default=CODEX_CLI_CMD)
    parser.add_argument("--skip-git-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    book_dir = os.path.abspath(args.book_dir)
    prompt_path = os.path.abspath(args.prompt_path)
    schema_path = os.path.abspath(args.schema_path)
    if not os.path.isdir(book_dir):
        print(f"Book directory not found: {book_dir}")
        return 1
    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        return 1
    if not os.path.exists(schema_path):
        print(f"Schema not found: {schema_path}")
        return 1

    source_md_path = find_source_markdown(book_dir)
    if source_md_path is None:
        print(f"No source Markdown found in: {book_dir}")
        return 1

    structured_root = os.path.join(book_dir, args.structured_dirname)
    progress_json_path = os.path.join(book_dir, args.progress_filename)
    log_path = os.path.join(book_dir, args.log_filename)
    manifest_path = os.path.join(book_dir, args.manifest_filename)
    inventory_path = os.path.join(book_dir, args.inventory_filename)

    manifest = {
        "book_dir": book_dir,
        "source_markdown": source_md_path,
        "structured_root": structured_root,
        "progress_json": progress_json_path,
        "prompt_path": prompt_path,
        "schema_path": schema_path,
        "generated_at": now_iso(),
    }
    write_manifest(manifest_path, manifest)

    progress = load_progress(progress_json_path, source_md_path, structured_root)
    log_event(log_path, f"run_start book_dir={book_dir} dry_run={args.dry_run}")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    lines = load_lines(source_md_path)
    headings = parse_headings(lines)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = find_git_root(script_dir) or find_git_root(book_dir)

    stages_bar = tqdm(total=4, desc="Preprocess Stages", unit="stage")

    if progress["stages"].get("structured_root_created"):
        stages_bar.update(1)
    else:
        os.makedirs(structured_root, exist_ok=True)
        progress["stages"]["structured_root_created"] = True
        progress["stages"]["structured_root_created_at"] = now_iso()
        update_progress(progress_json_path, progress)
        stages_bar.update(1)

    if progress["stages"].get("contents_extracted") and progress.get("sections"):
        stages_bar.update(1)
    else:
        prompt = build_prompt(prompt_path, source_md_path)
        add_dirs = [book_dir, script_dir] if CODEX_ADD_DIR else []
        toc_payload, error = run_codex_exec_json(
            args.codex_cmd,
            prompt,
            codex_cwd,
            add_dirs,
            CODEX_SKIP_GIT_CHECK or args.skip_git_check,
            schema_path,
            log_path,
        )
        if toc_payload is None:
            print(f"Preprocess failed during TOC extraction: {error}")
            return 1
        progress["book_title"] = toc_payload.get("book_title") or progress["book_title"]
        progress["sections"] = build_sections_from_toc(toc_payload)
        extract_contents_to_progress(lines, headings, progress, structured_root, log_path)
        progress["stages"]["contents_extracted"] = True
        progress["stages"]["contents_extracted_at"] = now_iso()
        update_progress(progress_json_path, progress)
        build_preprocess_inventory(progress, inventory_path)
        stages_bar.update(1)

    if progress["stages"].get("section_folders_created"):
        stages_bar.update(1)
    else:
        ensure_section_folders(progress, structured_root)
        progress["stages"]["section_folders_created"] = True
        progress["stages"]["section_folders_created_at"] = now_iso()
        update_progress(progress_json_path, progress)
        stages_bar.update(1)

    if progress["stages"].get("sections_extracted"):
        stages_bar.update(1)
    else:
        extract_sections(progress, lines, headings, structured_root, book_dir, log_path)
        write_atomic(
            os.path.join(structured_root, "FRONT-MATTER", "Contents", "Contents.md"),
            build_contents_navigation(progress, structured_root),
        )
        progress["stages"]["sections_extracted"] = all(section.get("extracted") for section in progress.get("sections", []))
        progress["stages"]["sections_extracted_at"] = now_iso()
        update_progress(progress_json_path, progress)
        build_preprocess_inventory(progress, inventory_path)
        stages_bar.update(1)

    stages_bar.close()
    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
