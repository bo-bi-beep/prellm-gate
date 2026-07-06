import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "tools" / "prellm_cost_benchmark.py"


spec = importlib.util.spec_from_file_location("prellm_cost_benchmark", HARNESS_PATH)
assert spec and spec.loader
benchmark = importlib.util.module_from_spec(spec)
sys.modules["prellm_cost_benchmark"] = benchmark
spec.loader.exec_module(benchmark)


class CostBenchmarkTests(unittest.TestCase):
    def test_parse_route_from_json(self) -> None:
        route, reason, parse_error = benchmark.parse_route('{"route":"rules","reason":"lookup"}')

        self.assertEqual(route, "rules")
        self.assertEqual(reason, "lookup")
        self.assertFalse(parse_error)

    def test_openai_base_url_accepts_v1_suffix(self) -> None:
        self.assertEqual(
            benchmark.openai_base_url("http://127.0.0.1:1234/v1"),
            "http://127.0.0.1:1234",
        )

    def test_usage_estimates_missing_provider_tokens(self) -> None:
        usage = benchmark.usage_from_provider(
            {},
            "prompt text",
            "output text",
            30.0,
            180.0,
        )

        self.assertTrue(usage.estimated)
        self.assertGreater(usage.total_tokens, 0)
        self.assertIsNotNone(usage.estimated_cost_usd)

    def test_codex_prompt_preserves_roles(self) -> None:
        prompt = benchmark.codex_prompt(
            [
                {"role": "system", "content": "Be careful."},
                {"role": "user", "content": "Fix the bug."},
            ]
        )

        self.assertIn("SYSTEM:\nBe careful.", prompt)
        self.assertIn("USER:\nFix the bug.", prompt)

    def test_codex_cli_provider_parses_usage_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile() as fake_codex:
            args = mock.Mock(
                codex_bin=fake_codex.name,
                codex_home="/tmp/codex-home",
                codex_ephemeral=True,
                codex_timeout_seconds=10,
            )
            stdout = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "t"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": "ok"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                        }
                    ),
                ]
            )
            completed = subprocess.CompletedProcess(args=["node"], returncode=0, stdout=stdout, stderr="")
            with mock.patch.object(benchmark.subprocess, "run", return_value=completed) as run:
                content, usage, _elapsed_ms = benchmark.call_codex_cli(
                    "gpt-5.5",
                    [{"role": "user", "content": "hello"}],
                    args,
                )

        self.assertEqual(content, "ok")
        self.assertEqual(usage["input_tokens"], 11)
        self.assertIn("--json", run.call_args.args[0])

    def test_fixture_dry_run_cli_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.json"
            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_PATH),
                    "--fixture-only",
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("rules_pre_gate", payload["strategies"])
            self.assertEqual(payload["strategies"]["rules_pre_gate"]["route_accuracy"], 1.0)
            self.assertIn("baseline_all_advanced", result.stdout)


if __name__ == "__main__":
    unittest.main()
