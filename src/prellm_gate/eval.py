from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import resources
import json
from time import perf_counter
from typing import Any

from .core import GateDecision, GateRequest, GateRoute, gate_request

DEFAULT_ROUTE_COSTS = {
    GateRoute.RULES.value: 0.01,
    GateRoute.CACHE.value: 0.01,
    GateRoute.CHEAP_MODEL.value: 0.08,
    GateRoute.CLARIFY.value: 0.02,
    GateRoute.EXPENSIVE_MODEL.value: 1.0,
}


@dataclass(frozen=True)
class ToyCase:
    name: str
    request: GateRequest
    expected_route: GateRoute


def _case_from_dict(item: dict[str, Any], default_context: dict[str, Any] | None = None) -> ToyCase:
    context = {**(default_context or {}), **item.get("context", {})}
    return ToyCase(
        name=item["name"],
        request=GateRequest(
            text=item["text"],
            context=context,
            history=item.get("history", []),
            trajectory=item.get("trajectory", []),
        ),
        expected_route=GateRoute(item["expected_route"]),
    )


def load_toy_cases() -> list[ToyCase]:
    raw = resources.files(__package__).joinpath("data/toy_cases.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    return [_case_from_dict(item) for item in payload["cases"]]


def _coding_case_from_dict(item: dict[str, Any]) -> ToyCase:
    text = item.get("text") or item.get("prompt") or item.get("instruction")
    if not isinstance(text, str):
        raise ValueError(f"coding case missing text/prompt/instruction: {item!r}")
    name = item.get("name") or item.get("task_id") or item.get("id") or "coding-case"
    context = {
        "domain": "coding",
        "benchmark": item.get("benchmark", "coding"),
    }
    expected_route = item.get("expected_route", GateRoute.EXPENSIVE_MODEL.value)
    return _case_from_dict(
        {
            "name": str(name),
            "text": text,
            "context": context,
            "expected_route": expected_route,
        }
    )


def load_coding_cases(jsonl_path: str | None = None) -> list[ToyCase]:
    if jsonl_path:
        cases: list[ToyCase] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cases.append(_coding_case_from_dict(json.loads(line)))
        return cases

    raw = resources.files(__package__).joinpath("data/coding_cases.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    return [_coding_case_from_dict(item) for item in payload["cases"]]


def _swebench_case_from_dict(item: dict[str, Any]) -> ToyCase:
    text = item.get("problem_statement") or item.get("text") or item.get("prompt")
    if not isinstance(text, str):
        raise ValueError(f"SWE-bench case missing problem_statement/text/prompt: {item!r}")
    name = item.get("instance_id") or item.get("name") or item.get("id") or "swebench-case"
    context = {
        "domain": "coding",
        "benchmark": "swebench",
        "repo": item.get("repo"),
    }
    expected_route = item.get("expected_route", GateRoute.EXPENSIVE_MODEL.value)
    return _case_from_dict(
        {
            "name": str(name),
            "text": text,
            "context": context,
            "expected_route": expected_route,
        }
    )


def load_swebench_cases(jsonl_path: str | None = None) -> list[ToyCase]:
    if jsonl_path:
        cases: list[ToyCase] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cases.append(_swebench_case_from_dict(json.loads(line)))
        return cases

    raw = resources.files(__package__).joinpath("data/swebench_cases.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    return [_swebench_case_from_dict(item) for item in payload["cases"]]


def _run_cases(
    cases: list[ToyCase],
    suite: str,
    route_costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    costs = {**DEFAULT_ROUTE_COSTS, **(route_costs or {})}
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    route_counts = {route.value: 0 for route in GateRoute}
    correct = 0
    false_deflections = 0
    estimated_cost = 0.0

    for case in cases:
        decision: GateDecision = gate_request(case.request)
        is_correct = decision.route == case.expected_route
        is_deflection = decision.route not in (GateRoute.EXPENSIVE_MODEL, GateRoute.CLARIFY)
        is_false_deflection = is_deflection and case.expected_route == GateRoute.EXPENSIVE_MODEL
        route_cost = costs[decision.route.value]
        correct += int(is_correct)
        false_deflections += int(is_false_deflection)
        route_counts[decision.route.value] += 1
        estimated_cost += route_cost
        rows.append(
            {
                "name": case.name,
                "expected": case.expected_route.value,
                "actual": decision.route.value,
                "route_cost": route_cost,
                "reason": decision.reason,
                "signals": decision.signals,
                "correct": is_correct,
                "false_deflection": is_false_deflection,
            }
        )

    elapsed_ms = (perf_counter() - started) * 1000
    total = len(cases)
    deflections = sum(route_counts[route.value] for route in (GateRoute.RULES, GateRoute.CACHE, GateRoute.CHEAP_MODEL))
    fallbacks = sum(route_counts[route.value] for route in (GateRoute.EXPENSIVE_MODEL, GateRoute.CLARIFY))
    baseline_cost = total * costs[GateRoute.EXPENSIVE_MODEL.value]
    cost_saved = baseline_cost - estimated_cost
    return {
        "suite": suite,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "deflection_rate": round(deflections / total, 3) if total else 0.0,
        "fallback_rate": round(fallbacks / total, 3) if total else 0.0,
        "false_deflections": false_deflections,
        "route_counts": route_counts,
        "route_costs": costs,
        "baseline_cost_units": round(baseline_cost, 3),
        "estimated_cost_units": round(estimated_cost, 3),
        "estimated_cost_saved_units": round(cost_saved, 3),
        "estimated_cost_savings_rate": round(cost_saved / baseline_cost, 3) if baseline_cost else 0.0,
        "elapsed_ms": round(elapsed_ms, 2),
        "avg_decision_ms": round(elapsed_ms / total, 4) if total else 0.0,
        "rows": rows,
    }


def run_toy_eval() -> dict[str, Any]:
    return _run_cases(load_toy_cases(), "toy")


def run_coding_eval(jsonl_path: str | None = None) -> dict[str, Any]:
    return _run_cases(load_coding_cases(jsonl_path), "coding")


def run_swebench_eval(jsonl_path: str | None = None) -> dict[str, Any]:
    return _run_cases(load_swebench_cases(jsonl_path), "swebench")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prellm-gate eval suites.")
    parser.add_argument(
        "--suite",
        choices=("toy", "coding", "swebench"),
        default="toy",
        help="Eval suite to run.",
    )
    parser.add_argument(
        "--jsonl",
        help="Optional JSONL benchmark file for coding or SWE-bench-style suites.",
    )
    args = parser.parse_args()

    if args.suite == "swebench":
        result = run_swebench_eval(args.jsonl)
    elif args.suite == "coding":
        result = run_coding_eval(args.jsonl)
    else:
        result = run_toy_eval()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
