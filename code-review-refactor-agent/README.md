# Code Review / Refactor Agent

A local-first CLI agent that scans a repository, asks an LLM to review it, and optionally produces a unified diff for refactoring.

## Features

- Review local repositories
- Generate Markdown code review reports
- Generate refactor patches in unified diff format
- Safe by default: does not modify files unless `--apply` is used
- Respects common ignored folders like `.git`, `node_modules`, `venv`, `dist`, and `build`
- Supports custom review focus, model name, file limits, and output paths

## Requirements

- Python 3.10+
- OpenAI Python SDK
- OpenAI API key

## Installation

```bash
python -m pip install -r requirements.txt
```

## Environment Variables

macOS / Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Optional model override:

```bash
export CODE_REVIEW_MODEL="gpt-5.2"
```

## Usage

Review a repository:

```bash
python code_review_agent.py . --mode review --focus "security, correctness"
```

Generate a refactor patch:

```bash
python code_review_agent.py . --mode refactor --focus "reduce duplication" --patch refactor.patch
```

Generate and apply a patch:

```bash
python code_review_agent.py . --mode refactor --focus "type hints and error handling" --apply
```

Print the report to the terminal as well as saving it:

```bash
python code_review_agent.py . --mode review --print
```

Choose a model:

```bash
python code_review_agent.py . --mode review --model gpt-5.2
```

## Output Files

By default, the agent writes:

- `code_review_report.md`
- `refactor.patch`, only in refactor mode when a patch is produced

These generated outputs are ignored by `.gitignore`.

## Safety

This tool does not overwrite source files by default.

In refactor mode, it writes a patch file first. If `--apply` is used, it validates the patch with `git apply --check` before applying it.

Always review generated patches before committing them.

## Uploading to GitHub from the Web UI

1. Create a new GitHub repository.
2. Open the repository page.
3. Click `Add file` → `Upload files`.
4. Upload these files:
   - `code_review_agent.py`
   - `README.md`
   - `.gitignore`
   - `requirements.txt`
   - `.env.example`
   - `LICENSE`
5. Click `Commit changes`.

## License

MIT
