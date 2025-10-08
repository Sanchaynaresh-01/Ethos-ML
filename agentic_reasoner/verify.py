from typing import List, Tuple
from .tools import NumericVerifier, ArithmeticVerifier, SchedulingVerifier, CubePaintVerifier, SequencePatternVerifier, GearTrainVerifier


def run_verification(problem: str, options: List[str], chosen_idx: int) -> Tuple[bool, str]:
    notes: List[str] = []
    overall_ok = True

    for V in (NumericVerifier, ArithmeticVerifier, SchedulingVerifier, CubePaintVerifier, SequencePatternVerifier, GearTrainVerifier):
        ok, note = V().verify(problem, options, chosen_idx)
        overall_ok = overall_ok and ok
        if note:
            notes.append(note)

    return overall_ok, " | ".join(notes)
