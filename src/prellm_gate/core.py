from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re


class GateRoute(str, Enum):
    RULES = "rules"
    CACHE = "cache"
    CHEAP_MODEL = "cheap-model"
    EXPENSIVE_MODEL = "expensive-model"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class GateRequest:
    text: str
    context: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    trajectory: list["TrajectoryStep"] = field(default_factory=list)


@dataclass(frozen=True)
class GateDecision:
    route: GateRoute
    reason: str
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryStep:
    action: str
    observation: str
    success: bool | None = None
    domain: str | None = None


_TRIVIAL_PATTERNS = (
    r"^what is the path for ",
    r"^what is the file path for ",
    r"^what's the status of ",
    r"^status check$",
    r"^yes or no: ",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _coerce_trajectory_step(item: Any) -> TrajectoryStep | None:
    if isinstance(item, TrajectoryStep):
        return item
    if not isinstance(item, dict):
        return None
    action = item.get("action")
    observation = item.get("observation")
    if not isinstance(action, str) or not isinstance(observation, str):
        return None
    success = item.get("success")
    if success is not None and not isinstance(success, bool):
        success = None
    domain = item.get("domain")
    return TrajectoryStep(
        action=action,
        observation=observation,
        success=success,
        domain=domain if isinstance(domain, str) else None,
    )


def _trajectory(request: GateRequest) -> list[TrajectoryStep]:
    raw_steps = [*request.trajectory]
    context_steps = request.context.get("trajectory")
    if isinstance(context_steps, list):
        raw_steps.extend(context_steps)
    steps = [_coerce_trajectory_step(item) for item in raw_steps]
    return [step for step in steps if step is not None]


def _trajectory_signals(request: GateRequest) -> dict[str, Any]:
    steps = _trajectory(request)[-5:]
    failed_steps = [
        step
        for step in steps
        if step.success is False
        or any(
            marker in _normalize(step.observation)
            for marker in ("error", "failed", "traceback", "permission denied", "unauthorized")
        )
    ]
    risky_actions = [
        step
        for step in steps
        if any(
            marker in _normalize(step.action)
            for marker in ("rm -rf", "git push", "deploy", "chmod 777", "curl | sh")
        )
    ]
    domains = sorted({step.domain for step in steps if step.domain})
    return {
        "trajectory_steps": len(steps),
        "failed_steps": len(failed_steps),
        "risky_actions": len(risky_actions),
        "domains": domains,
    }


def _is_coding_task(request: GateRequest) -> bool:
    domain = request.context.get("domain")
    benchmark = request.context.get("benchmark")
    if domain == "coding" or benchmark in {"humaneval", "mbpp", "coding"}:
        return True

    text = _normalize(request.text)
    coding_markers = (
        "write a function",
        "complete the function",
        "implement ",
        "debug ",
        "fix the bug",
        "unit test",
        "pytest",
        "assert ",
        "def ",
        "class ",
        "return ",
    )
    return any(marker in text for marker in coding_markers)


def _request_signals(request: GateRequest) -> dict[str, Any]:
    signals = _trajectory_signals(request)
    signals["coding_task"] = _is_coding_task(request)
    return signals


def _has_known_lookup(request: GateRequest) -> bool:
    known = request.context.get("known_paths")
    if not isinstance(known, dict):
        return False
    text = _normalize(request.text)
    return any(str(key).lower() in text for key in known)


def _is_repeated(request: GateRequest) -> bool:
    if not request.history:
        return False
    text = _normalize(request.text)
    return any(_normalize(item) == text for item in request.history[-3:])


def gate_request(request: GateRequest) -> GateDecision:
    text = _normalize(request.text)
    signals = _request_signals(request)

    if len(text) < 4:
        return GateDecision(GateRoute.CLARIFY, "request is too short to classify safely", 0.2, signals)

    if "?" not in text and len(text.split()) < 4:
        return GateDecision(GateRoute.CLARIFY, "underspecified request needs clarification", 0.3, signals)

    if signals["risky_actions"] or signals["failed_steps"] >= 2:
        return GateDecision(
            GateRoute.EXPENSIVE_MODEL,
            "recent environment trajectory has risky actions or repeated failures",
            0.86,
            signals,
        )

    if signals["coding_task"]:
        return GateDecision(
            GateRoute.EXPENSIVE_MODEL,
            "coding task should use a capable model unless a deterministic tool can fully handle it",
            0.82,
            signals,
        )

    if _has_known_lookup(request) or any(re.match(pattern, text) for pattern in _TRIVIAL_PATTERNS):
        return GateDecision(
            GateRoute.RULES,
            "matches a trivial lookup or deterministic pattern",
            0.95,
            signals,
        )

    if _is_repeated(request):
        return GateDecision(GateRoute.CACHE, "repeats a recent request", 0.92, signals)

    if len(text) < 32 and any(word in text for word in ("format", "rephrase", "summarize", "list")):
        return GateDecision(
            GateRoute.CHEAP_MODEL,
            "short request with low apparent reasoning load",
            0.63,
            signals,
        )

    if any(word in text for word in ("policy", "security", "exploit", "breach", "vulnerability")):
        return GateDecision(GateRoute.EXPENSIVE_MODEL, "higher-risk request should escalate", 0.88, signals)

    return GateDecision(GateRoute.EXPENSIVE_MODEL, "default to escalation when confidence is low", 0.5, signals)
