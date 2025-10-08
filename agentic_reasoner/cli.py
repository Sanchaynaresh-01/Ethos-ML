import argparse
import os
from typing import List

from .dataio import read_csv, write_output_csv
from .nb import MultinomialNB
from .agent import Agent
from .tools import ClassifierTool


MODEL_DIR_DEFAULT = "model"
MODEL_PATH = "model.json"


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _prepare_xy(rows):
    texts: List[str] = []
    labels: List[str] = []
    for r in rows:
        topic = r.get("topic", "")
        prob = r.get("problem_statement", "")
        opts = [
            r.get("answer_option_1", ""),
            r.get("answer_option_2", ""),
            r.get("answer_option_3", ""),
            r.get("answer_option_4", ""),
            r.get("answer_option_5", ""),
        ]
        combo = f"topic: {topic} \n problem: {prob} " + " ".join([f" opt{i+1}: {opts[i]}" for i in range(5)])
        lab = str(r.get("correct_option_number", "")).strip()
        if lab and lab in {"1", "2", "3", "4", "5"}:
            texts.append(combo)
            labels.append(lab)
    return texts, labels


def train_cmd(args: argparse.Namespace) -> None:
    rows = read_csv(args.train_csv)
    texts, labels = _prepare_xy(rows)
    model = MultinomialNB(alpha=1.0)
    model.fit(texts, labels)

    ensure_dir(args.model_dir)
    out_path = os.path.join(args.model_dir, MODEL_PATH)
    model.save(out_path)
    print(f"Model saved to {out_path}")


def predict_cmd(args: argparse.Namespace) -> None:
    in_rows = read_csv(args.test_csv)
    model_path = os.path.join(args.model_dir, MODEL_PATH)
    model = MultinomialNB.load(model_path)
    agent = Agent(ClassifierTool(model))

    out_rows = []
    jsonl_records = []
    for r in in_rows:
        topic = r.get("topic", "")
        prob = r.get("problem_statement", "")
        options = [
            r.get("answer_option_1", ""),
            r.get("answer_option_2", ""),
            r.get("answer_option_3", ""),
            r.get("answer_option_4", ""),
            r.get("answer_option_5", ""),
        ]
        choice, trace, details = agent.solve(topic, prob, options)
        out_rows.append((topic, prob, trace, choice))
        rec = {
            "topic": topic,
            "problem_statement": prob,
            "options": options,
            "choice": choice,
            "trace": trace,
            **details,
        }
        jsonl_records.append(rec)

    ensure_dir(os.path.dirname(args.out) or ".")
    write_output_csv(args.out, out_rows)
    print(f"Wrote predictions to {args.out}")

    if args.jsonl_out:
        ensure_dir(os.path.dirname(args.jsonl_out) or ".")
        import json
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL traces to {args.jsonl_out}")


def crossval_cmd(args: argparse.Namespace) -> None:
    import random
    rows = read_csv(args.train_csv)
    texts, labels = _prepare_xy(rows)
    n = len(texts)
    k = max(2, int(args.folds))
    idxs = list(range(n))
    random.Random(42).shuffle(idxs)
    folds = [idxs[i::k] for i in range(k)]

    def fit_on(idx_list):
        m = MultinomialNB(alpha=1.0)
        m.fit([texts[i] for i in idx_list], [labels[i] for i in idx_list])
        return m

    all_acc = []
    for fi in range(k):
        val_idx = set(folds[fi])
        train_idx = [i for i in range(n) if i not in val_idx]
        m = fit_on(train_idx)
        correct = 0
        total = 0
        for i in val_idx:
            pred, _ = m.predict(texts[i])
            if pred == labels[i]:
                correct += 1
            total += 1
        acc = (correct / total) if total else 0.0
        all_acc.append(acc)
        print(f"Fold {fi+1}/{k}: accuracy={acc:.4f} (n={total})")
    if all_acc:
        mean = sum(all_acc) / len(all_acc)
        print(f"Mean accuracy over {k} folds: {mean:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agentic Reasoning System CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("train_csv", help="Path to train.csv")
    p_train.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    p_train.set_defaults(func=train_cmd)

    p_pred = sub.add_parser("predict", help="Predict on test.csv and write output.csv")
    p_pred.add_argument("test_csv", help="Path to test.csv")
    p_pred.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    p_pred.add_argument("--out", default=os.path.join("artifacts", "output.csv"))
    p_pred.add_argument("--jsonl-out", default=os.path.join("artifacts", "output.jsonl"))
    p_pred.set_defaults(func=predict_cmd)

    p_cv = sub.add_parser("crossval", help="k-fold cross-validation on train.csv")
    p_cv.add_argument("train_csv", help="Path to train.csv")
    p_cv.add_argument("--folds", type=int, default=5)
    p_cv.set_defaults(func=crossval_cmd)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

import argparse
import os
from typing import List

from .dataio import read_csv, write_output_csv
from .nb import MultinomialNB
from .agent import Agent
from .tools import ClassifierTool


MODEL_DIR_DEFAULT = "model"
MODEL_PATH = "model.json"


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def train_cmd(args: argparse.Namespace) -> None:
    rows = read_csv(args.train_csv)
    texts: List[str] = []
    labels: List[str] = []
    for r in rows:
        topic = r.get("topic", "")
        prob = r.get("problem_statement", "")
        opts = [
            r.get("answer_option_1", ""),
            r.get("answer_option_2", ""),
            r.get("answer_option_3", ""),
            r.get("answer_option_4", ""),
            r.get("answer_option_5", ""),
        ]
        combo = f"topic: {topic} \n problem: {prob} " + " ".join([f" opt{i+1}: {opts[i]}" for i in range(5)])
        texts.append(combo)
        lab = str(r.get("correct_option_number", "")).strip()
        if lab and lab in {"1", "2", "3", "4", "5"}:
            labels.append(lab)
        else:
            # Skip malformed rows
            texts.pop()
    model = MultinomialNB(alpha=1.0)
    model.fit(texts, labels)

    ensure_dir(args.model_dir)
    out_path = os.path.join(args.model_dir, MODEL_PATH)
    model.save(out_path)
    print(f"Model saved to {out_path}")


def predict_cmd(args: argparse.Namespace) -> None:
    in_rows = read_csv(args.test_csv)
    model_path = os.path.join(args.model_dir, MODEL_PATH)
    model = MultinomialNB.load(model_path)
    agent = Agent(ClassifierTool(model))

    out_rows = []
    jsonl_records = []
    for r in in_rows:
        topic = r.get("topic", "")
        prob = r.get("problem_statement", "")
        options = [
            r.get("answer_option_1", ""),
            r.get("answer_option_2", ""),
            r.get("answer_option_3", ""),
            r.get("answer_option_4", ""),
            r.get("answer_option_5", ""),
        ]
        choice, trace, details = agent.solve(topic, prob, options)
        out_rows.append((topic, prob, trace, choice))
        rec = {
            "topic": topic,
            "problem_statement": prob,
            "options": options,
            "choice": choice,
            "trace": trace,
            **details,
        }
        jsonl_records.append(rec)

    ensure_dir(os.path.dirname(args.out) or ".")
    write_output_csv(args.out, out_rows)
    print(f"Wrote predictions to {args.out}")

    if args.jsonl_out:
        ensure_dir(os.path.dirname(args.jsonl_out) or ".")
        import json
        with open(args.jsonl_out, "w", encoding="utf-8") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL traces to {args.jsonl_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agentic Reasoning System CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("train_csv", help="Path to train.csv")
    p_train.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    p_train.set_defaults(func=train_cmd)

    p_pred = sub.add_parser("predict", help="Predict on test.csv and write output.csv")
    p_pred.add_argument("test_csv", help="Path to test.csv")
    p_pred.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    p_pred.add_argument("--out", default=os.path.join("artifacts", "output.csv"))
    p_pred.add_argument("--jsonl-out", default=os.path.join("artifacts", "output.jsonl"))
    p_pred.set_defaults(func=predict_cmd)

    p_cv = sub.add_parser("crossval", help="k-fold cross-validation on train.csv")
    p_cv.add_argument("train_csv", help="Path to train.csv")
    p_cv.add_argument("--folds", type=int, default=5)
    p_cv.set_defaults(func=crossval_cmd)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()