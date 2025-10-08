# Create artifacts directory
if (-not (Test-Path -LiteralPath "artifacts")) { New-Item -ItemType Directory -Path "artifacts" | Out-Null }

# Optional venv (not strictly required since project is pure-Python)
if (-not (Test-Path -LiteralPath ".venv")) { python -m venv .venv | Out-Null }

# Use venv's python if it exists
$python = if (Test-Path -LiteralPath ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }

# Train and predict
& $python -m agentic_reasoner.cli train "ML Challenge Dataset/train.csv" --model-dir model
& $python -m agentic_reasoner.cli crossval "ML Challenge Dataset/train.csv" --folds 5
& $python -m agentic_reasoner.cli predict "ML Challenge Dataset/test.csv" --model-dir model --out "artifacts/output.csv" --jsonl-out "artifacts/output.jsonl"

Write-Host "\nDone. Predictions with reasoning written to artifacts/output.csv and JSONL to artifacts/output.jsonl"
