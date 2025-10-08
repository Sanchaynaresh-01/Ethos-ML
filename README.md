# Agentic Reasoning System (Warp Ethos)

This project implements a lightweight, fully local Agentic Reasoning System designed for logic-based question answering with transparent reasoning traces.

It adheres to the challenge restrictions by:
- NOT using prohibited proprietary reasoning-heavy LLMs (no GPT-4/5, Claude Opus, Gemini Ultra, etc.)
- Using a custom, pure-Python Multinomial Naive Bayes classifier for option selection
- Incorporating a simple agentic pipeline: planning → tool selection → execution → verification → trace
- Adding lightweight symbolic/numeric tools (no external dependencies) for sanity checks and domain-specific verifiers (cube painting, sequences, gear trains, scheduling)

## Project Structure

- `agentic_reasoner/`
  - `agent.py` — Orchestrates planning, tool execution, verification, and tracing
  - `planner.py` — Breaks a problem into subtasks (decompose → select tool)
  - `tools.py` — Lightweight tools: classifier wrapper, numeric parser, simple schedulers
  - `nb.py` — Pure-Python Multinomial Naive Bayes model (train/save/load/infer)
  - `text.py` — Tokenization, normalization, n-gram extraction
  - `verify.py` — Verifiers to validate answers against parsed constraints
  - `dataio.py` — CSV I/O utilities (robust reader/writer)
  - `cli.py` — CLI entrypoint to train and run predictions
- `ML Challenge Dataset/` — Provided dataset (train.csv, test.csv, output.csv template)
- `run.ps1` — One-liner PowerShell script to train and generate predictions

## Quickstart (Windows PowerShell)

1) Train and predict in one go (uses the provided dataset):

```
pwsh path=null start=null
./run.ps1
```

This will:
- Create a virtual environment `.venv` (optional, used only for isolation)
- Train the model on `ML Challenge Dataset/train.csv`
- Generate CSV with reasoning traces at `./artifacts/output.csv`
- Generate JSONL traces at `./artifacts/output.jsonl`

2) Manual usage via CLI:

```
pwsh path=null start=null
python -m agentic_reasoner.cli train "ML Challenge Dataset/train.csv" --model-dir model
python -m agentic_reasoner.cli predict "ML Challenge Dataset/test.csv" --model-dir model --out "artifacts/output.csv" --jsonl-out "artifacts/output.jsonl"
```

3) Cross-validation:
```
pwsh path=null start=null
python -m agentic_reasoner.cli crossval "ML Challenge Dataset/train.csv" --folds 5
```

## Output Format
The predictor writes a CSV with the exact columns:
- `topic`
- `problem_statement`
- `solution` — a human-readable reasoning trace
- `correct option` — an integer 1–5, corresponding to the selected answer option

File path (by default): `artifacts/output.csv`

## Design Notes
- The classifier is intentionally simple to comply with restrictions. It learns token likelihoods per class (1–5) and produces a transparent trace listing the most influential tokens for the chosen option.
- The agent composes:
  - Decomposition plan (what subtasks to run)
  - Tool execution (classifier; numeric/time parsers when applicable)
  - Verification (e.g., if durations are parsed, ensure the chosen option is consistent)
- If verification disagrees with the classifier, the agent will log the conflict in the trace and still return the classifier’s best prediction (configurable), noting uncertainty.

## Reproducibility
- Pure Python, no compiled dependencies required
- Randomness is seeded

## Notes
- If you wish, you can extend `tools.py` with richer symbolic solvers (e.g., equation solving using your own simple parser) to cover more logic domains.