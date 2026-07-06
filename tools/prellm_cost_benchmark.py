from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from prellm_gate import GateRequest, GateRoute, gate_request


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_ADVANCED_PROVIDER = "openai"
DEFAULT_ADVANCED_MODEL = "gpt-5.5-pro"
DEFAULT_CODEX_BIN = (
    "/home/bibo/.openclaw-claw/npm/projects/openclaw-codex-8902d781d4/"
    "node_modules/@openclaw/codex/node_modules/@openai/codex/bin/codex.js"
)
DEFAULT_ADVANCED_INPUT_COST_PER_1M = 30.0
DEFAULT_ADVANCED_OUTPUT_COST_PER_1M = 180.0
ROUTES = {route.value for route in GateRoute}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    source: str
    text: str
    expected_route: str
    context: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool
    estimated_cost_usd: float | None
    cost_method: str


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, math.ceil(max(len(text) / 4, len(pieces) * 0.75)))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    return estimate_tokens("\n".join(f"{item['role']}: {item['content']}" for item in messages))


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> float | None:
    if input_cost_per_1m is None or output_cost_per_1m is None:
        return None
    return round((input_tokens / 1_000_000 * input_cost_per_1m) + (output_tokens / 1_000_000 * output_cost_per_1m), 8)


def usage_from_provider(
    usage: dict[str, Any] | None,
    prompt_text: str,
    output_text: str,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> TokenUsage:
    usage = usage or {}
    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_eval_count")
        or usage.get("eval_prompt_tokens")
    )
    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("eval_count")
        or usage.get("eval_completion_tokens")
    )
    estimated = input_tokens is None or output_tokens is None
    if input_tokens is None:
        input_tokens = estimate_tokens(prompt_text)
    if output_tokens is None:
        output_tokens = estimate_tokens(output_text)
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)

    provider_cost = usage.get("cost") or usage.get("total_cost") or usage.get("estimated_cost")
    if provider_cost is not None:
        cost = round(float(provider_cost), 8)
        cost_method = "provider_usage_cost"
    else:
        cost = compute_cost(input_tokens, output_tokens, input_cost_per_1m, output_cost_per_1m)
        cost_method = "configured_token_rates" if cost is not None else "not_available"

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated=estimated,
        estimated_cost_usd=cost,
        cost_method=cost_method,
    )


def dry_run_usage(
    prompt_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> TokenUsage:
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        estimated=True,
        estimated_cost_usd=compute_cost(prompt_tokens, output_tokens, input_cost_per_1m, output_cost_per_1m),
        cost_method="dry_run_estimate",
    )


def fetch_swebench_rows(dataset: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
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


def built_in_easy_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="easy-known-path",
            source="builtin-easy",
            text="What is the path for config?",
            expected_route=GateRoute.RULES.value,
            context={"known_paths": {"config": "config/app.yaml"}},
        ),
        BenchmarkCase(
            case_id="easy-repeated-status",
            source="builtin-easy",
            text="Status check?",
            expected_route=GateRoute.CACHE.value,
            history=["Status check?", "Status check?"],
        ),
        BenchmarkCase(
            case_id="easy-rephrase",
            source="builtin-easy",
            text="Please rephrase this line.",
            expected_route=GateRoute.CHEAP_MODEL.value,
        ),
        BenchmarkCase(
            case_id="easy-clarify",
            source="builtin-easy",
            text="Help",
            expected_route=GateRoute.CLARIFY.value,
        ),
        BenchmarkCase(
            case_id="easy-architecture-reasoning",
            source="builtin-easy",
            text="How should this architecture handle fallback uncertainty?",
            expected_route=GateRoute.EXPENSIVE_MODEL.value,
        ),
    ]


def swebench_cases(rows: list[dict[str, Any]]) -> list[BenchmarkCase]:
    cases = []
    for row in rows:
        cases.append(
            BenchmarkCase(
                case_id=str(row["instance_id"]),
                source="swebench-lite",
                text=str(row["problem_statement"]),
                expected_route=GateRoute.EXPENSIVE_MODEL.value,
                context={"domain": "coding", "benchmark": "swebench", "repo": row.get("repo")},
            )
        )
    return cases


def advanced_messages(case: BenchmarkCase) -> list[dict[str, str]]:
    if case.source == "swebench-lite":
        repo = case.context.get("repo") or "unknown"
        user = (
            "This is a SWE-bench Lite issue-fixing task. Produce a concise patch plan and the most likely "
            "files/functions to inspect. Do not claim tests passed unless evidence is provided.\n\n"
            f"instance_id: {case.case_id}\nrepo: {repo}\nproblem_statement:\n{case.text}"
        )
    else:
        user = f"Answer the user request concisely.\n\nrequest:\n{case.text}"
    return [
        {"role": "system", "content": "You are a careful coding-agent assistant."},
        {"role": "user", "content": user},
    ]


def gate_messages(case: BenchmarkCase) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict pre-LLM gate. Return only JSON with keys route and reason. "
                "Allowed routes: rules, cache, cheap-model, expensive-model, clarify. "
                "Use expensive-model for repository issue fixing, patch generation, debugging, "
                "or tests that need real code reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"case_id: {case.case_id}\nsource: {case.source}\n"
                f"context: {json.dumps(case.context, sort_keys=True)}\nrequest:\n{case.text}"
            ),
        },
    ]


def parse_route(text: str) -> tuple[str | None, str, bool]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            route = payload.get("route")
            if route in ROUTES:
                return str(route), str(payload.get("reason", "")), False
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    for route in ROUTES:
        if route in lowered:
            return route, text[:240], True
    return None, text[:240], True


def request_json(url: str, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def openai_base_url(url: str) -> str:
    return url[:-3].rstrip("/") if url.rstrip("/").endswith("/v1") else url.rstrip("/")


def call_advanced_model(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], float]:
    if provider == "codex-cli":
        return call_codex_cli(model, messages, args)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --execute --advanced-provider openai is set")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when --execute --advanced-provider openrouter is set")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/bo-bi-beep/prellm-gate",
            "X-Title": "prellm-gate cost benchmark",
        }
    else:
        raise RuntimeError(f"unsupported advanced provider: {provider}")
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    payload = request_json(url, body=body, headers=headers)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return payload["choices"][0]["message"]["content"], payload.get("usage") or {}, elapsed_ms


def codex_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


def call_codex_cli(
    model: str,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], float]:
    codex_bin = Path(args.codex_bin)
    if not codex_bin.exists():
        raise RuntimeError(f"Codex CLI entrypoint does not exist: {codex_bin}")

    command = [
        "node",
        str(codex_bin),
        "-a",
        "never",
        "exec",
        "--json",
        "--model",
        model,
        "--sandbox",
        "read-only",
        "--cd",
        str(Path.cwd()),
    ]
    if args.codex_ephemeral:
        command.append("--ephemeral")
    command.append(codex_prompt(messages))

    env = os.environ.copy()
    if args.codex_home:
        env["CODEX_HOME"] = args.codex_home

    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.codex_timeout_seconds,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    content = ""
    usage: dict[str, Any] = {}
    errors: list[str] = []
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                content = str(item.get("text") or content)
        elif event_type == "turn.completed":
            usage = event.get("usage") or usage
        elif event_type in {"error", "turn.failed"}:
            error = event.get("message") or (event.get("error") or {}).get("message")
            if error:
                errors.append(str(error))

    if process.returncode != 0:
        details = "; ".join(errors) or process.stderr.strip()[:500] or f"exit status {process.returncode}"
        quoted = shlex.join(command[:8] + ["..."])
        raise RuntimeError(f"Codex CLI provider failed via {quoted}: {details}")

    return content, usage, elapsed_ms


def discover_local_endpoint(base_url: str | None, local_model: str | None, kind: str) -> dict[str, str] | None:
    candidates: list[tuple[str, str]] = []
    if base_url:
        candidates.append((base_url.rstrip("/"), kind))
    for env_name in ("PRELLM_GATE_LOCAL_URL", "LOCAL_LLM_BASE_URL", "OLLAMA_HOST"):
        value = os.environ.get(env_name)
        if value:
            candidates.append((value.rstrip("/"), kind))
    candidates.extend(
        [
            ("http://127.0.0.1:11434", "ollama"),
            ("http://localhost:11434", "ollama"),
            ("http://127.0.0.1:11435", "llamacpp"),
            ("http://localhost:11435", "llamacpp"),
            ("http://desktop-161rhm9.tail8684a5.ts.net:11435", "llamacpp"),
            ("http://127.0.0.1:1234", "openai"),
            ("http://localhost:1234", "openai"),
            ("http://127.0.0.1:8080", "openai"),
            ("http://localhost:8080", "openai"),
        ]
    )

    seen: set[tuple[str, str]] = set()
    for url, candidate_kind in candidates:
        if candidate_kind == "auto":
            for inferred in ("ollama", "openai", "llamacpp"):
                endpoint = discover_local_endpoint(url, local_model, inferred)
                if endpoint:
                    return endpoint
            continue
        key = (url, candidate_kind)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate_kind == "ollama":
                payload = request_json(f"{url}/api/tags")
                models = [item.get("name") or item.get("model") for item in payload.get("models", [])]
            elif candidate_kind == "llamacpp":
                payload = request_json(f"{url}/v1/models")
                models = [item.get("id") for item in payload.get("data", [])]
            else:
                payload = request_json(f"{openai_base_url(url)}/v1/models")
                models = [item.get("id") for item in payload.get("data", [])]
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            continue
        model = select_local_model([item for item in models if isinstance(item, str)], local_model)
        if model:
            return {"base_url": url, "kind": candidate_kind, "model": model}
    return None


def select_local_model(models: list[str], requested: str | None) -> str | None:
    if requested:
        return requested if not models or requested in models else None
    for marker in ("gemma", "llama"):
        for model in models:
            if marker in model.lower():
                return model
    return models[0] if models else None


def call_local_gate(endpoint: dict[str, str], messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any], float]:
    started = time.perf_counter()
    if endpoint["kind"] == "ollama":
        payload = request_json(
            f"{endpoint['base_url']}/api/chat",
            body={
                "model": endpoint["model"],
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            headers={"Content-Type": "application/json"},
        )
        content = payload.get("message", {}).get("content", "")
        usage = {
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "eval_count": payload.get("eval_count"),
        }
    elif endpoint["kind"] == "llamacpp":
        prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
        payload = request_json(
            f"{endpoint['base_url']}/completion",
            body={
                "prompt": prompt,
                "temperature": 0,
                "n_predict": max_tokens,
            },
            headers={"Content-Type": "application/json"},
        )
        content = payload.get("content", "")
        timings = payload.get("timings") or {}
        usage = {
            "prompt_tokens": payload.get("tokens_evaluated") or timings.get("prompt_n"),
            "completion_tokens": payload.get("tokens_predicted") or timings.get("predicted_n"),
        }
    else:
        payload = request_json(
            f"{openai_base_url(endpoint['base_url'])}/v1/chat/completions",
            body={
                "model": endpoint["model"],
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            headers={"Content-Type": "application/json"},
        )
        content = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
    elapsed_ms = (time.perf_counter() - started) * 1000
    return content, usage, elapsed_ms


def zero_usage() -> TokenUsage:
    return TokenUsage(0, 0, 0, False, 0.0, "zero")


def add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    if left.estimated_cost_usd is None or right.estimated_cost_usd is None:
        cost = None
    else:
        cost = round(left.estimated_cost_usd + right.estimated_cost_usd, 8)
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated=left.estimated or right.estimated,
        estimated_cost_usd=cost,
        cost_method="aggregate",
    )


def run_advanced_call(
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages = advanced_messages(case)
    prompt_text = "\n".join(message["content"] for message in messages)
    if args.execute:
        content, raw_usage, elapsed_ms = call_advanced_model(
            args.advanced_provider,
            args.advanced_model,
            messages,
            args.max_advanced_tokens,
            args.temperature,
            args,
        )
        usage = usage_from_provider(
            raw_usage,
            prompt_text,
            content,
            args.advanced_input_cost_per_1m,
            args.advanced_output_cost_per_1m,
        )
    else:
        content = "DRY_RUN_NO_PROVIDER_CALL"
        raw_usage = {}
        elapsed_ms = 0.0
        usage = dry_run_usage(
            estimate_messages_tokens(messages),
            args.dry_run_advanced_output_tokens,
            args.advanced_input_cost_per_1m,
            args.advanced_output_cost_per_1m,
        )
    return {
        "called": True,
        "model": args.advanced_model,
        "elapsed_ms": round(elapsed_ms, 2),
        "usage": asdict(usage),
        "raw_usage": raw_usage,
        "output_preview": content[:500],
    }


def run_baseline(cases: list[BenchmarkCase], args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    total_usage = zero_usage()
    for case in cases:
        advanced = run_advanced_call(case, args)
        usage = TokenUsage(**advanced["usage"])
        total_usage = add_usage(total_usage, usage)
        rows.append(row_result(case, GateRoute.EXPENSIVE_MODEL.value, "baseline always calls advanced model", zero_usage(), advanced))
    return strategy_summary("baseline_all_advanced", rows, total_usage, zero_usage())


def run_rules_gate(cases: list[BenchmarkCase], args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    advanced_total = zero_usage()
    gate_total = zero_usage()
    for case in cases:
        started = time.perf_counter()
        decision = gate_request(
            GateRequest(
                text=case.text,
                context=case.context,
                history=case.history,
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        gate_usage = TokenUsage(0, 0, 0, False, 0.0, "deterministic_local_rules")
        gate_total = add_usage(gate_total, gate_usage)
        advanced = skipped_advanced()
        if decision.route == GateRoute.EXPENSIVE_MODEL:
            advanced = run_advanced_call(case, args)
            advanced_total = add_usage(advanced_total, TokenUsage(**advanced["usage"]))
        rows.append(
            row_result(
                case,
                decision.route.value,
                decision.reason,
                gate_usage,
                advanced,
                gate_elapsed_ms=round(elapsed_ms, 4),
                gate_signals=decision.signals,
            )
        )
    return strategy_summary("rules_pre_gate", rows, advanced_total, gate_total)


def run_local_llm_gate(cases: list[BenchmarkCase], args: argparse.Namespace) -> dict[str, Any]:
    endpoint = None if args.skip_local_gate else discover_local_endpoint(args.local_base_url, args.local_model, args.local_kind)
    rows = []
    advanced_total = zero_usage()
    gate_total = zero_usage()
    for case in cases:
        messages = gate_messages(case)
        prompt_text = "\n".join(message["content"] for message in messages)
        raw_usage: dict[str, Any] = {}
        if endpoint and args.execute:
            content, raw_usage, elapsed_ms = call_local_gate(endpoint, messages, args.max_gate_tokens)
            usage = usage_from_provider(
                raw_usage,
                prompt_text,
                content,
                args.local_input_cost_per_1m,
                args.local_output_cost_per_1m,
            )
        else:
            content = '{"route":"expensive-model","reason":"dry run or local endpoint unavailable; fail closed"}'
            elapsed_ms = 0.0
            usage = dry_run_usage(
                estimate_messages_tokens(messages),
                args.dry_run_gate_output_tokens,
                args.local_input_cost_per_1m,
                args.local_output_cost_per_1m,
            )

        route, reason, parse_error = parse_route(content)
        if route is None:
            route = GateRoute.EXPENSIVE_MODEL.value
            reason = "local gate response did not parse; fail closed to advanced model"
            parse_error = True
        gate_total = add_usage(gate_total, usage)
        advanced = skipped_advanced()
        if route == GateRoute.EXPENSIVE_MODEL.value:
            advanced = run_advanced_call(case, args)
            advanced_total = add_usage(advanced_total, TokenUsage(**advanced["usage"]))
        rows.append(
            row_result(
                case,
                route,
                reason,
                usage,
                advanced,
                gate_elapsed_ms=round(elapsed_ms, 2),
                gate_raw_usage=raw_usage,
                gate_model=endpoint,
                gate_parse_error=parse_error,
            )
        )
    summary = strategy_summary("local_cheap_llm_pre_gate", rows, advanced_total, gate_total)
    summary["local_endpoint"] = endpoint
    summary["local_gate_available"] = endpoint is not None
    return summary


def skipped_advanced() -> dict[str, Any]:
    return {
        "called": False,
        "model": None,
        "elapsed_ms": 0.0,
        "usage": asdict(zero_usage()),
        "raw_usage": {},
        "output_preview": "",
    }


def row_result(
    case: BenchmarkCase,
    actual_route: str,
    reason: str,
    gate_usage: TokenUsage,
    advanced: dict[str, Any],
    gate_elapsed_ms: float = 0.0,
    gate_signals: dict[str, Any] | None = None,
    gate_raw_usage: dict[str, Any] | None = None,
    gate_model: dict[str, str] | None = None,
    gate_parse_error: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source": case.source,
        "expected_route": case.expected_route,
        "actual_route": actual_route,
        "route_correct": actual_route == case.expected_route,
        "false_deflection": actual_route != GateRoute.EXPENSIVE_MODEL.value and case.expected_route == GateRoute.EXPENSIVE_MODEL.value,
        "gate": {
            "reason": reason,
            "elapsed_ms": gate_elapsed_ms,
            "usage": asdict(gate_usage),
            "raw_usage": gate_raw_usage or {},
            "model": gate_model,
            "parse_error": gate_parse_error,
            "signals": gate_signals or {},
        },
        "advanced": advanced,
    }


def strategy_summary(name: str, rows: list[dict[str, Any]], advanced_total: TokenUsage, gate_total: TokenUsage) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["route_correct"])
    advanced_calls = sum(1 for row in rows if row["advanced"]["called"])
    false_deflections = sum(1 for row in rows if row["false_deflection"])
    total_usage = add_usage(advanced_total, gate_total)
    baseline_note = "cost baseline; route score treats all tasks as expensive-model" if name == "baseline_all_advanced" else ""
    return {
        "name": name,
        "score_kind": "route_correctness_proxy",
        "score_note": baseline_note or "route correctness only; answer quality and patch resolution are not evaluated",
        "total_cases": total,
        "route_correct": correct,
        "route_accuracy": round(correct / total, 3) if total else 0.0,
        "false_deflections": false_deflections,
        "advanced_calls": advanced_calls,
        "advanced_call_rate": round(advanced_calls / total, 3) if total else 0.0,
        "advanced_usage": asdict(advanced_total),
        "gate_usage": asdict(gate_total),
        "total_usage": asdict(total_usage),
        "rows": rows,
    }


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases = built_in_easy_cases()
    if args.fixture_only:
        return cases
    rows = fetch_swebench_rows(args.dataset, args.split, args.offset, args.swebench_length)
    return [*cases, *swebench_cases(rows)]


def compare_to_baseline(summary: dict[str, Any]) -> None:
    baseline_cost = summary["strategies"]["baseline_all_advanced"]["total_usage"]["estimated_cost_usd"]
    baseline_calls = summary["strategies"]["baseline_all_advanced"]["advanced_calls"]
    for name, strategy in summary["strategies"].items():
        cost = strategy["total_usage"]["estimated_cost_usd"]
        if baseline_cost is None or cost is None:
            strategy["estimated_cost_saved_vs_baseline_usd"] = None
            strategy["estimated_cost_savings_rate_vs_baseline"] = None
        else:
            saved = round(baseline_cost - cost, 8)
            strategy["estimated_cost_saved_vs_baseline_usd"] = saved
            strategy["estimated_cost_savings_rate_vs_baseline"] = round(saved / baseline_cost, 3) if baseline_cost else 0.0
        calls_saved = baseline_calls - strategy["advanced_calls"]
        strategy["advanced_calls_saved_vs_baseline"] = calls_saved
        strategy["advanced_call_savings_rate_vs_baseline"] = round(calls_saved / baseline_calls, 3) if baseline_calls else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all-advanced, rules-gated, and local-LLM-gated prellm cost.")
    parser.add_argument("--execute", action="store_true", help="Make live provider calls. Omitted means dry-run estimates only.")
    parser.add_argument("--fixture-only", action="store_true", help="Use only built-in mixed cases; skip SWE-bench Lite fetch.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--swebench-length", type=int, default=3)
    parser.add_argument("--advanced-provider", choices=("openai", "openrouter", "codex-cli"), default=DEFAULT_ADVANCED_PROVIDER)
    parser.add_argument("--advanced-model", default=DEFAULT_ADVANCED_MODEL)
    parser.add_argument("--advanced-input-cost-per-1m", type=float, default=DEFAULT_ADVANCED_INPUT_COST_PER_1M)
    parser.add_argument("--advanced-output-cost-per-1m", type=float, default=DEFAULT_ADVANCED_OUTPUT_COST_PER_1M)
    parser.add_argument("--local-input-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--local-output-cost-per-1m", type=float, default=0.0)
    parser.add_argument("--local-base-url")
    parser.add_argument("--local-model")
    parser.add_argument("--local-kind", choices=("auto", "ollama", "openai", "llamacpp"), default="auto")
    parser.add_argument("--skip-local-gate", action="store_true")
    parser.add_argument("--max-advanced-tokens", type=int, default=800)
    parser.add_argument("--max-gate-tokens", type=int, default=120)
    parser.add_argument("--dry-run-advanced-output-tokens", type=int, default=256)
    parser.add_argument("--dry-run-gate-output-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--codex-bin", default=DEFAULT_CODEX_BIN)
    parser.add_argument("--codex-home", help="Optional CODEX_HOME for ChatGPT/OAuth-backed codex exec runs.")
    parser.add_argument("--codex-timeout-seconds", type=int, default=180)
    parser.add_argument("--codex-ephemeral", action="store_true", help="Run codex exec without persisting session files.")
    parser.add_argument("--output", default="artifacts/prellm_cost_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    summary = {
        "benchmark": "prellm-gate exact cost comparison",
        "mode": "live_execute" if args.execute else "dry_run_estimate",
        "score_kind": "route correctness and token/cost accounting proxy",
        "score_caveat": (
            "This harness does not check out repos, apply patches, or run tests; it must not be reported as "
            "SWE-bench patch-resolution."
        ),
        "case_sources": {
            "builtin_easy_cases": len([case for case in cases if case.source == "builtin-easy"]),
            "swebench_lite_cases": len([case for case in cases if case.source == "swebench-lite"]),
            "swebench_dataset": args.dataset,
            "swebench_split": args.split,
            "swebench_offset": args.offset,
        },
        "pricing": {
            "advanced_provider": args.advanced_provider,
            "advanced_model": args.advanced_model,
            "advanced_input_cost_per_1m": args.advanced_input_cost_per_1m,
            "advanced_output_cost_per_1m": args.advanced_output_cost_per_1m,
            "local_input_cost_per_1m": args.local_input_cost_per_1m,
            "local_output_cost_per_1m": args.local_output_cost_per_1m,
        },
        "strategies": {},
    }
    try:
        summary["strategies"] = {
            "baseline_all_advanced": run_baseline(cases, args),
            "rules_pre_gate": run_rules_gate(cases, args),
            "local_cheap_llm_pre_gate": run_local_llm_gate(cases, args),
        }
        compare_to_baseline(summary)
    except urllib.error.HTTPError as exc:
        summary["blocked"] = {
            "where": "provider_http_request",
            "http_status": exc.code,
            "body": exc.read().decode("utf-8", errors="replace")[:500],
        }
    except RuntimeError as exc:
        summary["blocked"] = {
            "where": "runtime_configuration",
            "message": str(exc),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    compact = {}
    if summary["strategies"]:
        compact = {
            name: {
                "route_accuracy": strategy["route_accuracy"],
                "false_deflections": strategy["false_deflections"],
                "advanced_calls": strategy["advanced_calls"],
                "advanced_tokens": strategy["advanced_usage"]["total_tokens"],
                "gate_tokens": strategy["gate_usage"]["total_tokens"],
                "estimated_cost_usd": strategy["total_usage"]["estimated_cost_usd"],
                "estimated_cost_saved_vs_baseline_usd": strategy["estimated_cost_saved_vs_baseline_usd"],
            }
            for name, strategy in summary["strategies"].items()
        }
    if summary.get("blocked"):
        compact["blocked"] = summary["blocked"]
    print(json.dumps(compact, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
