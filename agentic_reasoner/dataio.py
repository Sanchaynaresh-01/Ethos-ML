import csv
from typing import Dict, Iterable, List, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def write_output_csv(path: str, rows: Iterable[Tuple[str, str, str, int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "problem_statement", "solution", "correct option"])
        for r in rows:
            writer.writerow(list(r))