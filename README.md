# prellm-gate

Cheap pre-inference gating for trivial agent calls.

`prellm-gate` is a small router that decides whether a request should be handled by rules, cache, retrieval, a cheap model, or an expensive model.

## Paper

This prototype implements the routing idea inspired by:

- **AgentWorld: An Interactive Simulation Platform for Agentic AI**
- arXiv: https://arxiv.org/abs/2606.24597

The paper frames agent work as environment trajectories: actions, observations, tools, domains, and outcomes.
`prellm-gate` applies that framing before inference: inspect cheap request and trajectory signals first, then decide whether the next step deserves a deterministic handler, cache hit, cheap local model, or stronger advanced model.

## Why this exists

Coding agents and tool-heavy workflows waste latency, money, and energy on requests that do not need full inference.
This project explores a cheap pre-LLM gate that catches obvious trivia, repeats, and low-risk requests before they hit expensive models.

## What it does

- Classifies requests before model invocation
- Routes trivial work to rules, cache, or retrieval
- Allows a cheap local LLM to act as a gate before escalating hard work
- Reads recent action/observation trajectory signals when available
- Escalates only when uncertainty or task value is high enough
- Records decisions so threshold choices can be measured instead of guessed

## What it is not

- Not a new foundation model
- Not a planner replacement
- Not a general chatbot framework
- Not a hidden second agent stack

## Decision paths

- `rules` for exact-match or deterministic responses
- `cache` for repeated requests or known outputs
- `cheap-model` for ambiguous but low-value cases, including a small local LLM gate
- `expensive-model` for real reasoning or high-risk tasks
- `clarify` when the request is underspecified

## Minimal API

```python
from prellm_gate import GateRequest, gate_request

request = GateRequest(
    text="What is the path for the config file?",
    context={"known_paths": {"config": "config/app.yaml"}},
)

decision = gate_request(request)
print(decision.route)
print(decision.reason)
```

## Trajectory signals

AgentWorld frames agent work as environment trajectories: action/observation pairs across domains such as terminal, web, search, OS, Android, SWE, and MCP.
`prellm-gate` keeps that idea small and operational.
If recent trajectory steps show repeated failures or risky actions, the gate escalates even when the current text looks simple.

```python
from prellm_gate import GateRequest, TrajectoryStep, gate_request

request = GateRequest(
    text="Can we just retry the same command?",
    trajectory=[
        TrajectoryStep(
            domain="terminal",
            action="python deploy.py",
            observation="Permission denied",
            success=False,
        ),
        TrajectoryStep(
            domain="terminal",
            action="python deploy.py --force",
            observation="failed with exit status 1",
            success=False,
        ),
    ],
)

decision = gate_request(request)
print(decision.route)
print(decision.signals)
```

## Toy evaluation

The repository includes a tiny benchmark of trivial, ambiguous, and high-value requests.
The goal is not to win on accuracy alone, but to measure:

- deflection rate
- fallback rate
- false deflections
- route mix
- decision latency
- estimated route cost versus an all-advanced-model baseline

Run it with:

```bash
PYTHONPATH=src python3 -m prellm_gate.eval
```

## Coding eval

The coding suite is a small HumanEval/MBPP-style routing benchmark.
Its first pass is intentionally conservative: code synthesis, bug fixing, and test generation should escalate instead of being deflected to rules or cache.

```bash
PYTHONPATH=src python3 -m prellm_gate.eval --suite coding
```

You can also point the coding suite at an external JSONL file with common benchmark fields such as `task_id` plus `prompt`, `text`, or `instruction`.
Unless a row provides `expected_route`, the coding loader expects `expensive-model`.

```bash
PYTHONPATH=src python3 -m prellm_gate.eval --suite coding --jsonl path/to/benchmark.jsonl
```

## SWE-bench-style eval

The SWE-bench-style suite treats repository issue-fixing tasks as high-value coding work.
For now the bundled cases are synthetic, but the loader accepts external JSONL with fields such as `instance_id`, `repo`, and `problem_statement`.
The important metrics are both quality and economics:

- `false_deflections`: cases that should have gone to the advanced model but were routed cheaply
- `elapsed_ms` and `avg_decision_ms`: routing overhead
- `estimated_cost_units`: route-cost estimate for the gate policy
- `estimated_cost_savings_rate`: savings against an all-advanced-model baseline

```bash
PYTHONPATH=src python3 -m prellm_gate.eval --suite swebench
PYTHONPATH=src python3 -m prellm_gate.eval --suite swebench --jsonl path/to/swebench.jsonl
```

## MVP scope

- one synchronous gate API
- one rule-based classifier
- one fallback model path
- one toy evaluation runner, one coding benchmark-style runner, and one SWE-bench-style runner
- one test suite for core routing behavior

## Open questions

- What is the cheapest reliable signal set before inference?
- When should ambiguous requests clarify instead of escalating?
- Which thresholds are stable across domains?
- How much routing logic is too much before the gate becomes its own agent?

## Roadmap

1. Local prototype
2. Add a benchmark harness with realistic examples
3. Add observability for routing outcomes
4. Compare rules-only, hybrid, and cheap-model gate variants
5. Add real SWE-bench Lite/Verified adapters and cost traces
