#!/usr/bin/env python3
"""
Code Review / Refactor Agent

A safe, local-first CLI agent that scans a repository, asks an LLM to review it,
and optionally produces a unified diff for refactoring. It never overwrites files
unless you pass --apply, and even then it validates the patch with `git apply --check`.

Install:
    python -m pip install -r requirements.txt

Set your key:
    export OPENAI_API_KEY="sk-..."

Examples:
    python code_review_agent.py . --mode review --focus "security, correctness"
    python code_review_agent.py . --mode refactor --focus "reduce duplication" --patch refactor.patch
    python code_review_agent.py . --mode refactor --focus "type hints and error handling" --apply
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - user-facing error path
    OpenAI = None  # type: ignore[assignment]


DEFAULT_MODEL = os.getenv("CODE_REVIEW_MODEL", "gpt-5.2")

DEFAULT_INCLUDE_EXTENSIONS = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".kts",
    ".lua",
    ".md",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

DEFAULT_INCLUDE_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Rakefile",
    "Gemfile",
    "Pipfile",
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
    "webpack.config.js",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    "target",
    ".next",
    ".nuxt",
    ".cache",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "*.lock",
    "*.min.js",
    "*.map",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.gz",
    "*.7z",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.class",
    "*.jar",
    "*.wasm",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "uv.lock",
    "poetry.lock",
}

LANGUAGE_BY_EXTENSION = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".ini": "ini",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".php": "php",
    ".ps1": "powershell",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".scss": "scss",
    ".sh": "bash",
    ".sql": "sql",
    ".svelte": "svelte",
    ".swift": "swift",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "zsh",
}


@dataclasses.dataclass(frozen=True)
class FileItem:
    rel_path: str
    abs_path: Path
    text: str
    language: str
    byte_size: int


@dataclasses.dataclass(frozen=True)
class Snippet:
    rel_path: str
    language: str
    text: str
    part_index: int = 1
    part_count: int = 1

    @property
    def label(self) -> str:
        if self.part_count <= 1:
            return self.rel_path
        return f"{self.rel_path}  [part {self.part_index}/{self.part_count}]"


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    root: Path
    mode: str
    focus: str
    model: str
    output: Path
    patch: Path
    apply: bool
    print_report: bool
    max_files: int
    max_file_bytes: int
    chunk_chars: int
    max_output_tokens: int
    include_extensions: set[str]
    include_filenames: set[str]
    exclude_dirs: set[str]
    exclude_patterns: set[str]
    respect_gitignore: bool


def normalize_extensions(raw: str | None) -> set[str]:
    if not raw:
        return set(DEFAULT_INCLUDE_EXTENSIONS)
    values: set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.add(item if item.startswith(".") else f".{item}")
    return values


def normalize_csv_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def load_gitignore_patterns(root: Path) -> set[str]:
    patterns: set[str] = set()
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return patterns
    try:
        for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            patterns.add(line.rstrip("/"))
            patterns.add(line)
    except OSError:
        pass
    return patterns


def matches_any_pattern(rel_path: str, name: str, patterns: Iterable[str]) -> bool:
    normalized = rel_path.replace(os.sep, "/")
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        bare = pattern.rstrip("/")
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(normalized, pattern):
            return True
        if fnmatch.fnmatch(normalized, f"**/{pattern}"):
            return True
        if bare and (normalized == bare or normalized.startswith(f"{bare}/")):
            return True
    return False


def should_include_file(path: Path, include_extensions: set[str], include_filenames: set[str]) -> bool:
    return path.name in include_filenames or path.suffix.lower() in include_extensions


def is_probably_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    if not data:
        return False
    textish = bytes(range(32, 127)) + b"\n\r\t\f\b"
    non_text = sum(byte not in textish for byte in data[:4096])
    return non_text / max(1, min(len(data), 4096)) > 0.30


def language_for(path: Path) -> str:
    if path.name == "Dockerfile":
        return "dockerfile"
    if path.name == "Makefile":
        return "makefile"
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "text")


def collect_files(config: AgentConfig) -> list[FileItem]:
    root = config.root.resolve()
    patterns = set(config.exclude_patterns)
    if config.respect_gitignore:
        patterns |= load_gitignore_patterns(root)

    collected: list[FileItem] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_dir = "." if current == root else current.relative_to(root).as_posix()

        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            rel = dirname if rel_dir == "." else f"{rel_dir}/{dirname}"
            if dirname in config.exclude_dirs:
                continue
            if matches_any_pattern(rel, dirname, patterns):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            abs_path = current / filename
            rel_path = abs_path.relative_to(root).as_posix()

            if not should_include_file(abs_path, config.include_extensions, config.include_filenames):
                continue
            if matches_any_pattern(rel_path, filename, patterns):
                continue

            try:
                data = abs_path.read_bytes()
            except OSError:
                continue

            if len(data) > config.max_file_bytes:
                data = data[: config.max_file_bytes]
                truncated_note = "\n\n[TRUNCATED: file exceeded max file byte limit]\n"
            else:
                truncated_note = ""

            if is_probably_binary(data):
                continue

            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")

            collected.append(
                FileItem(
                    rel_path=rel_path,
                    abs_path=abs_path,
                    text=text + truncated_note,
                    language=language_for(abs_path),
                    byte_size=len(data),
                )
            )

            if len(collected) >= config.max_files:
                return collected

    return collected


def make_project_tree(files: Sequence[FileItem], max_lines: int = 300) -> str:
    lines = [item.rel_path for item in files]
    if len(lines) > max_lines:
        visible = lines[:max_lines]
        visible.append(f"... ({len(lines) - max_lines} more files omitted from tree)")
        lines = visible
    return "\n".join(f"- {line}" for line in lines)


def make_snippets(file: FileItem, max_chars_per_snippet: int) -> list[Snippet]:
    if len(file.text) <= max_chars_per_snippet:
        return [Snippet(file.rel_path, file.language, file.text)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in file.text.splitlines(keepends=True):
        if current and current_len + len(line) > max_chars_per_snippet:
            chunks.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("".join(current))

    total = len(chunks)
    return [
        Snippet(
            rel_path=file.rel_path,
            language=file.language,
            text=chunk,
            part_index=index + 1,
            part_count=total,
        )
        for index, chunk in enumerate(chunks)
    ]


def chunk_snippets(files: Sequence[FileItem], chunk_chars: int) -> list[list[Snippet]]:
    snippets: list[Snippet] = []
    for file in files:
        snippets.extend(make_snippets(file, max(8_000, chunk_chars // 2)))

    chunks: list[list[Snippet]] = []
    current: list[Snippet] = []
    current_len = 0

    for snippet in snippets:
        rendered_len = len(snippet.text) + len(snippet.label) + 80
        if current and current_len + rendered_len > chunk_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(snippet)
        current_len += rendered_len

    if current:
        chunks.append(current)
    return chunks


def render_snippets(snippets: Sequence[Snippet]) -> str:
    blocks: list[str] = []
    for snippet in snippets:
        blocks.append(
            f"### {snippet.label}\n"
            f"```{snippet.language}\n"
            f"{snippet.text}\n"
            f"```"
        )
    return "\n\n".join(blocks)


def build_instructions(mode: str) -> str:
    shared = """
You are a senior code review and refactoring agent.
Be precise, practical, and evidence-based. Prefer concrete findings over generic advice.
Do not invent files, functions, dependencies, or test results that are not visible in the provided context.
When you are unsure, say so and explain what extra context would prove it.
Prioritize issues by severity: Critical, High, Medium, Low.
Focus on correctness, security, maintainability, performance, tests, observability, and API design.
""".strip()

    if mode == "review":
        return shared + """

Return a Markdown report with these sections:
1. Executive summary
2. Top risks and bugs
3. Security review
4. Maintainability and design
5. Performance notes
6. Testing gaps
7. Suggested refactor plan
8. File-by-file notes

For each important issue include:
- Severity
- File/path
- Why it matters
- Suggested fix
"""

    return shared + """

You are in REFACTOR mode.
Return a Markdown report plus minimal unified diffs in fenced ```diff blocks.
Patch rules:
- Only change files shown in the context.
- Keep patches small and reviewable.
- Do not include placeholder code.
- Do not introduce new third-party dependencies unless explicitly justified.
- Prefer behavior-preserving refactors unless the focus asks for bug fixes.
- Unified diff headers must use paths relative to the repository root.
- If a safe patch cannot be produced from the available context, explain why instead of inventing one.

Markdown sections:
1. Refactor summary
2. Risks addressed
3. Patch
4. Manual follow-ups
"""


def build_chunk_prompt(
    *,
    config: AgentConfig,
    project_tree: str,
    snippets: Sequence[Snippet],
    chunk_index: int,
    chunk_count: int,
) -> str:
    focus = config.focus or "general correctness, security, maintainability, and testability"
    return f"""
Repository root: {config.root}
Mode: {config.mode}
Focus: {focus}
Chunk: {chunk_index}/{chunk_count}

Project tree:
{project_tree}

Review only the files in this chunk, but consider the project tree for context.

Files in this chunk:
{render_snippets(snippets)}
""".strip()


def build_synthesis_prompt(config: AgentConfig, chunk_reports: Sequence[str]) -> str:
    joined = "\n\n".join(
        f"## Chunk report {index + 1}\n\n{report}" for index, report in enumerate(chunk_reports)
    )
    if config.mode == "review":
        task = """
Synthesize the chunk reports into one final, deduplicated Markdown code review.
Keep only findings that are well-supported by the provided reports.
Group issues by severity and include actionable fixes.
""".strip()
    else:
        task = """
Synthesize the chunk reports into one final, deduplicated refactor report.
Keep all valid unified diff blocks exactly as patch blocks; remove duplicate or conflicting patches.
If patches conflict or are risky, call that out clearly.
""".strip()
    return f"{task}\n\nFocus: {config.focus}\n\n{joined}"


def require_openai_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("Missing dependency: run `python -m pip install -r requirements.txt`.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def call_model(client: "OpenAI", *, model: str, instructions: str, prompt: str, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                pieces.append(str(text))
    return "\n".join(pieces).strip()


def extract_diff_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    for match in pattern.finditer(markdown):
        block = match.group(1).strip("\n")
        if re.search(r"(?m)^diff --git ", block) or re.search(r"(?m)^---\s+", block):
            blocks.append(block)
    return blocks


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_git_apply(root: Path, patch_path: Path) -> tuple[bool, str]:
    check = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if check.returncode != 0:
        return False, "Patch validation failed:\n" + (check.stderr or check.stdout)

    apply = subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if apply.returncode != 0:
        return False, "Patch application failed:\n" + (apply.stderr or apply.stdout)
    return True, "Patch applied successfully. Review `git diff` before committing."


def run_agent(config: AgentConfig) -> tuple[str, str | None]:
    files = collect_files(config)
    if not files:
        raise RuntimeError("No reviewable files found. Try adjusting --include-ext or --max-files.")

    project_tree = make_project_tree(files)
    chunks = chunk_snippets(files, config.chunk_chars)
    client = require_openai_client()
    instructions = build_instructions(config.mode)

    print(f"Found {len(files)} files. Reviewing in {len(chunks)} chunk(s)...", file=sys.stderr)

    chunk_reports: list[str] = []
    for index, snippets in enumerate(chunks, start=1):
        print(f"Processing chunk {index}/{len(chunks)}...", file=sys.stderr)
        prompt = build_chunk_prompt(
            config=config,
            project_tree=project_tree,
            snippets=snippets,
            chunk_index=index,
            chunk_count=len(chunks),
        )
        report = call_model(
            client,
            model=config.model,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=config.max_output_tokens,
        )
        chunk_reports.append(report)

    if len(chunk_reports) == 1:
        final_report = chunk_reports[0]
    else:
        print("Synthesizing final report...", file=sys.stderr)
        final_report = call_model(
            client,
            model=config.model,
            instructions=instructions,
            prompt=build_synthesis_prompt(config, chunk_reports),
            max_output_tokens=config.max_output_tokens,
        )

    metadata = textwrap.dedent(
        f"""
        <!--
        Generated by code_review_agent.py
        Mode: {config.mode}
        Model: {config.model}
        Root: {config.root.resolve()}
        Files reviewed: {len(files)}
        Chunks: {len(chunks)}
        Focus: {config.focus or 'general'}
        -->
        """
    ).strip()
    final_report = metadata + "\n\n" + final_report.strip() + "\n"

    patch_text: str | None = None
    if config.mode == "refactor":
        diff_blocks: list[str] = []
        for report in [*chunk_reports, final_report]:
            diff_blocks.extend(extract_diff_blocks(report))
        if diff_blocks:
            seen: set[str] = set()
            unique_blocks: list[str] = []
            for block in diff_blocks:
                if block not in seen:
                    unique_blocks.append(block)
                    seen.add(block)
            patch_text = "\n\n".join(unique_blocks).rstrip() + "\n"

    return final_report, patch_text


def parse_args(argv: Sequence[str]) -> AgentConfig:
    parser = argparse.ArgumentParser(
        description="Code Review / Refactor Agent powered by the OpenAI Responses API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=".", help="Repository or directory to review.")
    parser.add_argument("--mode", choices={"review", "refactor"}, default="review", help="Agent mode.")
    parser.add_argument("--focus", default="", help="Review/refactor focus, e.g. 'security and error handling'.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--output", default="code_review_report.md", help="Markdown report output path.")
    parser.add_argument("--patch", default="refactor.patch", help="Patch output path for refactor mode.")
    parser.add_argument("--apply", action="store_true", help="Apply generated patch with git apply after validation.")
    parser.add_argument("--print", dest="print_report", action="store_true", help="Also print the report to stdout.")
    parser.add_argument("--max-files", type=int, default=250, help="Maximum files to review.")
    parser.add_argument("--max-file-bytes", type=int, default=120_000, help="Maximum bytes read per file.")
    parser.add_argument("--chunk-chars", type=int, default=60_000, help="Approximate characters per model request.")
    parser.add_argument("--max-output-tokens", type=int, default=8_000, help="Maximum output tokens per model call.")
    parser.add_argument(
        "--include-ext",
        default="",
        help="Comma-separated extensions to include. Empty means common code/config extensions.",
    )
    parser.add_argument(
        "--include-name",
        default="",
        help="Comma-separated exact filenames to include in addition to extension matches.",
    )
    parser.add_argument(
        "--exclude-dir",
        default="",
        help="Comma-separated directory names to exclude in addition to defaults.",
    )
    parser.add_argument(
        "--exclude-pattern",
        default="",
        help="Comma-separated fnmatch patterns to exclude in addition to defaults.",
    )
    parser.add_argument(
        "--no-gitignore",
        dest="respect_gitignore",
        action="store_false",
        help="Do not apply simple .gitignore pattern filtering.",
    )
    parser.set_defaults(respect_gitignore=True)

    args = parser.parse_args(argv)
    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        parser.error(f"Path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Path must be a directory: {root}")
    if args.apply and args.mode != "refactor":
        parser.error("--apply only makes sense with --mode refactor")

    include_names = set(DEFAULT_INCLUDE_FILENAMES) | normalize_csv_set(args.include_name)
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | normalize_csv_set(args.exclude_dir)
    exclude_patterns = set(DEFAULT_EXCLUDE_PATTERNS) | normalize_csv_set(args.exclude_pattern)

    return AgentConfig(
        root=root,
        mode=args.mode,
        focus=args.focus.strip(),
        model=args.model,
        output=Path(args.output).expanduser(),
        patch=Path(args.patch).expanduser(),
        apply=bool(args.apply),
        print_report=bool(args.print_report),
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        chunk_chars=args.chunk_chars,
        max_output_tokens=args.max_output_tokens,
        include_extensions=normalize_extensions(args.include_ext),
        include_filenames=include_names,
        exclude_dirs=exclude_dirs,
        exclude_patterns=exclude_patterns,
        respect_gitignore=bool(args.respect_gitignore),
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    try:
        config = parse_args(argv)
        report, patch_text = run_agent(config)
        write_text(config.output, report)
        print(f"Report written to: {config.output}", file=sys.stderr)

        if config.print_report:
            print(report)

        if config.mode == "refactor":
            if patch_text:
                write_text(config.patch, patch_text)
                print(f"Patch written to: {config.patch}", file=sys.stderr)
                if config.apply:
                    ok, message = run_git_apply(config.root, config.patch)
                    print(message, file=sys.stderr)
                    return 0 if ok else 2
            else:
                print("No valid unified diff blocks were produced.", file=sys.stderr)

        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
