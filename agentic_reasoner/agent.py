import os
import random
from typing import Dict, List, Tuple

from .planner import make_plan
from .tools import ClassifierTool
from .verify import run_verification
from .text import normalize_whitespace

random.seed(42)


def combine_text(topic: str, problem: str, options: List[str]) -> str:
    # Combine fields to feed the classifier; include option text so the model can learn option-specific cues
    buff = [f"topic: {topic}", f"problem: {problem}"]
    for i, opt in enumerate(options, 1):
        buff.append(f"opt{i}: {opt}")
    return " \n ".join(buff)


def build_trace(plan_steps, cls_explanation, verifier_ok: bool, verifier_note: str, chosen: int, probs: Dict[str, float]) -> str:
    lines: List[str] = []
    lines.append("Plan:")
    for s in plan_steps:
        lines.append(f" - {s.name}: {s.detail}")
    lines.append("")
    lines.append("Execution:")
    prob_str = ", ".join([f"{k}={probs[k]:.3f}" for k in sorted(probs.keys())])
    lines.append(f" - Classifier probabilities: {prob_str}")
    if cls_explanation:
        lines.append(" - Top contributing tokens for chosen option:")
        for tok, val in cls_explanation:
            lines.append(f"    - '{tok}' -> {val:.3f}")
    lines.append("")
    lines.append("Verification:")
    lines.append(f" - Result: {'PASS' if verifier_ok else 'WARN'} | {verifier_note}")
    lines.append("")
    lines.append(f"Decision: Option {chosen}")
    return "\n".join(lines)


class Agent:
    def __init__(self, clf_tool: ClassifierTool):
        self.clf_tool = clf_tool

    def solve(self, topic: str, problem: str, options: List[str]):
        plan = make_plan(topic, problem)
        combo = combine_text(topic, problem, options)
        label, probs = self.clf_tool.score(combo)
        chosen = int(label)
        explanation = self.clf_tool.explain(combo, label, k=8)
        ok, note = run_verification(problem, options, chosen)
        trace = build_trace(plan.steps, explanation, ok, note, chosen, probs)
        details = {
            "plan": [
                {"name": s.name, "detail": s.detail} for s in plan.steps
            ],
            "probs": probs,
            "explanation": [{"token": t, "weight": float(w)} for t, w in explanation],
            "verification": {"ok": ok, "note": note},
            "chosen": chosen,
        }
        return chosen, normalize_whitespace(trace), details
