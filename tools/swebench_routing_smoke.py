from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from prellm_gate import GateRequest, GateRoute, gate_request


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_ADVANCED_MODEL = "openai/gpt-5.5-pro"
DEFAULT_CHEAP_MODEL = "openai/gpt-5.4-mini"


def fetch_rows(dataset: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{HF_ROWS_URL}?{params}", timeout=30) as response:
        payload = json.load(response)
    return [item["row"] for item in payload["rows"]]


def expected_route(_: dict[str, Any]) -> str:
    return GateRoute.EXPENSIVE_MODEL.value


def prompt_for(row: dict[str, Any]) -> str:
    return (
        "Classify this software engineering benchmark task for a pre-LLM gate.\n"
        "Return only one JSON object with keys route and reason.\n"
        "Allowed routes: rules, cache, cheap-model, expensive-model, clarify.\n"
        "Use expensive-model for repository issue fixing, patch generation, debugging, "
        "or tests that need real code reasoning.\n\n"
        f"instance_id: {row['instance_id']}\n"
        f"repo: {row['repo']}\n"
        f"problem_statement:\n{row['problem_statement']}\n"
    )


def parse_route(text: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            route = payload.get("route")
            if route in {item.value for item in GateRoute}:
                return route, str(payload.get("reason", ""))
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    for route in GateRoute:
        if route.value in lowered:
            return route.value, text[:200]
    return "parse-error", text[:200]


def call_openrouter(model: str, row: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for model router strategies")

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict routing classifier for model escalation decisions.",
            },
            {"role": "user", "content": prompt_for(row)},
        ],
        "temperature": 0,
        "max_tokens": 120,
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bo-bi-beep/prellm-gate",
            "X-Title": "prellm-gate swebench smoke",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    elapsed_ms = (time.perf_counter() - started) * 1000
    content = payload["choices"][0]["message"]["content"]
    route, reason = parse_route(content)
    return {
        "route": route,
        "reason": reason,
        "usage": payload.get("usage") or {},
        "elapsed_ms": round(elapsed_ms, 2),
    }


def run_model_strategy(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    correct = 0

    for row in rows:
        result = call_openrouter(model, row)
        expected = expected_route(row)
        is_correct = result["route"] == expected
        correct += int(is_correct)
        usage = result["usage"]
        prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        results.append(
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "expected": expected,
                "actual": result["route"],
                "correct": is_correct,
                "usage": usage,
                "elapsed_ms": result["elapsed_ms"],
            }
        )

    return {
        "model": model,
        "score": f"{correct}/{len(rows)}",
        "accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "rows": results,
    }


def run_deterministic_strategy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    correct = 0
    started = time.perf_counter()

    for row in rows:
        decision = gate_request(
            GateRequest(
                text=row["problem_statement"],
                context={"domain": "coding", "benchmark": "swebench", "repo": row["repo"]},
            )
        )
        expected = expected_route(row)
        is_correct = decision.route.value == expected
        correct += int(is_correct)
        results.append(
            {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "expected": expected,
                "actual": decision.route.value,
                "correct": is_correct,
                "reason": decision.reason,
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "model": "deterministic-prellm-gate",
        "score": f"{correct}/{len(rows)}",
        "accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_ms": round(elapsed_ms, 2),
        "rows": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small SWE-bench Lite routing/token smoke test.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--advanced-model", default=DEFAULT_ADVANCED_MODEL)
    parser.add_argument("--cheap-model", default=DEFAULT_CHEAP_MODEL)
    parser.add_argument("--output", default="artifacts/swebench_lite_routing_smoke.json")
    args = parser.parse_args()

    rows = fetch_rows(args.dataset, args.split, args.offset, args.length)
    summary = {
        "dataset": f"{args.dataset} {args.split} rows offset={args.offset} length={args.length}",
        "task": "routing only: expected route is expensive-model for SWE-bench issue-fixing tasks",
        "strategies": {
            "best_model_router": run_model_strategy(args.advanced_model, rows),
            "cheap_llm_pre_gate": run_model_strategy(args.cheap_model, rows),
            "deterministic_prellm_gate": run_deterministic_strategy(rows),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: _summary(item) for name, item in summary["strategies"].items()}, indent=2))
    print(f"wrote {output}")


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": result["model"],
        "score": result["score"],
        "accuracy": result["accuracy"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
    }


if __name__ == "__main__":
    main()
