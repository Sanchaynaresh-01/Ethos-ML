import re
from typing import Dict, List, Optional, Tuple

from .nb import MultinomialNB

# Simple numeric/time parser
DUR_RE = re.compile(r"(?:(\d+(?:\.\d+)?)\s*(hours?|hrs?|h))|(?:(\d+(?:\.\d+)?)\s*(minutes?|mins?|m))", re.I)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_minutes(text: str) -> float:
    total = 0.0
    for m in DUR_RE.finditer(text):
        if m.group(1):
            total += float(m.group(1)) * 60
        if m.group(3):
            total += float(m.group(3))
    return total


def has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text))


def extract_numbers(text: str) -> List[float]:
    return [float(x) for x in NUM_RE.findall(text)]


class ClassifierTool:
    def __init__(self, model: MultinomialNB):
        self.model = model

    def score(self, text: str) -> Tuple[str, Dict[str, float]]:
        return self.model.predict(text)

    def explain(self, text: str, cls: str, k: int = 8) -> List[Tuple[str, float]]:
        return self.model.top_features_for(text, cls, k=k)


class NumericVerifier:
    def verify(self, problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
        # Only attempt very simple consistency checks on durations
        if not has_numbers(problem):
            return True, "No numeric data to verify."
        # Summarize per-option durations if any
        notes = []
        for i, opt in enumerate(options, 1):
            opt_mins = parse_minutes(opt)
            if opt_mins > 0:
                notes.append(f"Option {i} mentions ~{int(opt_mins)} minutes")
        ok = True
        return ok, ("; ".join(notes) or "No per-option durations")


class ArithmeticVerifier:
    """Very lightweight arithmetic sanity checks based on extracted numbers.
    - Summarize totals in problem vs per-option numeric mentions.
    - Never forces a failure; only annotates potential matches.
    """

    def verify(self, problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
        p_nums = extract_numbers(problem)
        if not p_nums:
            return True, "No arithmetic signals."
        # Heuristic: compute simple aggregates
        total = sum(p_nums)
        avg = total / max(1, len(p_nums))
        notes = [f"Problem numbers sum≈{total:g}, avg≈{avg:g}"]
        # See which options mention a number near total
        for i, opt in enumerate(options, 1):
            nums = extract_numbers(opt)
            if not nums:
                continue
            # nearest number to total
            nearest = min(nums, key=lambda x: abs(x - total))
            if abs(nearest - total) <= max(1.0, 0.05 * max(1.0, abs(total))):
                notes.append(f"Option {i} mentions ≈problem-sum ({nearest:g})")
        return True, "; ".join(notes)


class SchedulingVerifier:
    """Detect simple machine-time problems like A/B/C with per-item durations and horizon.
    Provides a conservative note about throughput; typically order doesn't change steady-state throughput.
    """

    TIME_PER_ITEM_RE = re.compile(r"\b([ABC])\b[^\d]*?(\d+(?:\.\d+)?)\s*(minutes?|mins?|m)\b", re.I)
    HORIZON_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|m)\b", re.I)

    def parse_machine_times(self, text: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for m in self.TIME_PER_ITEM_RE.finditer(text):
            mach = m.group(1).upper()
            mins = float(m.group(2))
            out[mach] = mins
        return out

    def parse_horizon_minutes(self, text: str) -> Optional[float]:
        # Choose the largest minutes mention as horizon when phrased like "in 60 minutes"
        candidates = [float(m.group(1)) for m in self.HORIZON_RE.finditer(text)]
        if not candidates:
            return None
        return max(candidates)

    def compute_pipeline_items(self, order: List[str], times: Dict[str, float], horizon: float) -> int:
        t_seq = [times[x] for x in order]
        total = sum(t_seq)
        max_t = max(t_seq)
        if horizon < total:
            return 0
        # First completion after total, then one every max_t
        return 1 + int((horizon - total) // max_t)

    def extract_order_from_option(self, opt: str) -> Optional[List[str]]:
        # Expect patterns like "A -> B -> C"
        parts = [p.strip().upper() for p in opt.split("->")]
        parts = [p for p in parts if p in {"A", "B", "C"}]
        return parts if len(parts) >= 3 else None

    def verify(self, problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
        times = self.parse_machine_times(problem)
        horizon = self.parse_horizon_minutes(problem)
        if not times or horizon is None:
            return True, "No scheduling structure detected."
        notes = [f"Detected machine minutes: {times}, horizon≈{int(horizon)}m"]
        best_items = -1
        best_opts: List[int] = []
        for i, opt in enumerate(options, 1):
            order = self.extract_order_from_option(opt)
            if not order:
                continue
            items = self.compute_pipeline_items(order, times, horizon)
            notes.append(f"Option {i} ('{'->'.join(order)}') completes≈{items}")
            if items > best_items:
                best_items = items
                best_opts = [i]
            elif items == best_items:
                best_opts.append(i)
        if best_items >= 0:
            notes.append(f"Best throughput option(s): {best_opts}")
            if chosen_idx not in best_opts:
                notes.append(f"Chosen {chosen_idx} not among best; keeping classifier decision but flagging.")
                return True, "; ".join(notes)
        return True, "; ".join(notes)


class CubePaintVerifier:
    """Recognize N×N×N painted cube problems and compute count of small cubes with exactly two painted faces.
    For an N×N×N cube painted on all outside faces: exactly-two-painted count = 12*(N-2) for N>=2.
    Also supports inferring N from total count (e.g., 27 => N=3).
    """

    NXN_RE = re.compile(r"(\d+)\s*[x×]\s*(\d+)\s*[x×]\s*(\d+)", re.I)
    TOTAL_CUBES_RE = re.compile(r"\b(\d+)\b\s*(?:cubes|smaller cubes|small cubes)", re.I)

    def infer_n(self, text: str) -> Optional[int]:
        # Try explicit N×N×N
        m = self.NXN_RE.search(text)
        if m:
            a, b, c = map(int, m.groups())
            if a == b == c:
                return a
        # Try total cube count being a perfect cube
        for m2 in self.TOTAL_CUBES_RE.finditer(text):
            tot = int(m2.group(1))
            n = round(tot ** (1/3))
            if n*n*n == tot:
                return n
        return None

    def two_face_count(self, n: int) -> int:
        if n < 2:
            return 0
        return 12 * max(0, n - 2)

    def verify(self, problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
        n = self.infer_n(problem)
        if not n:
            return True, "No cube structure detected."
        count = self.two_face_count(n)
        notes = [f"Detected N={n}, two-painted count={count}"]
        # Flag options that mention count exactly
        hits = []
        for i, opt in enumerate(options, 1):
            nums = extract_numbers(opt)
            if any(abs(x - count) < 1e-9 for x in nums):
                hits.append(i)
        if hits:
            notes.append(f"Options mentioning exact count: {hits}")
        if hits and chosen_idx not in hits:
            notes.append(f"Chosen {chosen_idx} not among exact-count options; keeping classifier decision but flagging.")
        return True, "; ".join(notes)


class SequencePatternVerifier:
    """Attempt to infer next (or next two) numbers from sequences using difference and second-difference heuristics.
    If options contain matching numbers, annotate the best ones.
    """

    SEQ_RE = re.compile(r"(?:(?:^|\D))(?:-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)(?:\D|$)")

    def parse_sequence(self, text: str) -> Optional[List[float]]:
        m = self.SEQ_RE.search(text)
        if not m:
            return None
        frag = m.group(0)
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", frag)]
        return nums if len(nums) >= 3 else None

    def next_numbers(self, seq: List[float], k: int = 1) -> List[float]:
        # Use constant second-difference if applicable; else use last difference repeating.
        diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        if len(diffs) >= 2:
            dd = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            if all(abs(x - dd[0]) < 1e-9 for x in dd):
                last = diffs[-1]
                step = dd[0]
                out = []
                cur = seq[-1]
                for _ in range(k):
                    last += step
                    cur += last
                    out.append(cur)
                return out
        # Fallback: repeat last difference
        last_diff = diffs[-1]
        out = []
        cur = seq[-1]
        for _ in range(k):
            cur += last_diff
            out.append(cur)
        return out

    def verify(self, problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
        seq = self.parse_sequence(problem)
        if not seq:
            return True, "No sequence pattern detected."
        # Determine if question asks for next one or next two
        ask_two = bool(re.search(r"next\s*(two|2)|two\s*numbers|sixth\s*and\s*seventh", problem, re.I))
        preds = self.next_numbers(seq, k=2 if ask_two else 1)
        notes = [f"Sequence parsed={seq}", f"Predicted next{' two' if ask_two else ''}={preds}"]
        hits = []
        for i, opt in enumerate(options, 1):
            nums = extract_numbers(opt)
            if not nums:
                continue
            if ask_two:
                if len(nums) >= 2 and abs(nums[0] - preds[0]) < 1e-9 and abs(nums[1] - preds[1]) < 1e-9:
                    hits.append(i)
            else:
                if any(abs(x - preds[0]) < 1e-9 for x in nums):
                    hits.append(i)
        if hits:
            notes.append(f"Option(s) matching predicted value(s): {hits}")
            if chosen_idx not in hits:
                notes.append(f"Chosen {chosen_idx} not among predicted-match options; keeping classifier decision but flagging.")
        return True, " | ".join(notes)


class GearTrainVerifier:
    """Parse gear train with A,B,C teeth counts and predict direction and rotations for C when A turns once.
    Result: rotations_C = teeth_A / teeth_C, direction: same as A if 2 meshes (A-B, B-C) => even flips.
    """

    TEETH_RE = re.compile(r"Gear\s*A\s*has\s*(\d+)\s*teeth.*?Gear\s*B\s*has\s*(\d+)\s*teeth.*?Gear\s*C\s*has\s*(\d+)\s*teeth", re.I | re.S)

    def verify(self, problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
        m = self.TEETH_RE.search(problem)
        if not m:
            return True, "No gear train detected."
        a, b, c = map(int, m.groups())
        rotations_c = a / c if c else 0.0
        direction = "clockwise"  # two meshes -> same as A
        notes = [f"Gear teeth A={a}, B={b}, C={c}; predicted C≈{rotations_c:g} rotations {direction}"]
        # If options include numeric near rotations or contain direction keyword, note hits
        hits = []
        for i, opt in enumerate(options, 1):
            nums = extract_numbers(opt)
            dir_hit = re.search(r"clockwise|counter[- ]?clockwise|anticlockwise", opt, re.I)
            num_hit = any(abs(x - rotations_c) <= 0.05 for x in nums)  # allow 0.05 tolerance
            if num_hit or dir_hit:
                hits.append(i)
        if hits and chosen_idx not in hits:
            notes.append(f"Options referencing computed result: {hits}; chosen {chosen_idx} differs (FYI only).")
        return True, "; ".join(notes)
