from dataclasses import dataclass
from typing import List

@dataclass
class PlanStep:
    name: str
    detail: str

@dataclass
class Plan:
    steps: List[PlanStep]


def make_plan(topic: str, problem_statement: str) -> Plan:
    steps: List[PlanStep] = []
    # Very lightweight heuristic planning
    steps.append(PlanStep(name="ParseProblem", detail="Analyze text, extract signals (numbers, keywords)") )
    steps.append(PlanStep(name="SelectTool", detail="Pick classifier; enable numeric verification if numbers present") )
    steps.append(PlanStep(name="ExecuteTool", detail="Run classifier to score each option (1..5)") )
    steps.append(PlanStep(name="Verify", detail="If possible, sanity-check with numeric/time parsing") )
    steps.append(PlanStep(name="AssembleTrace", detail="Compose transparent reasoning trace") )
    return Plan(steps=steps)