import unittest

from prellm_gate import GateRequest, GateRoute, TrajectoryStep, gate_request
from prellm_gate.eval import run_coding_eval, run_swebench_eval, run_toy_eval


class GateTests(unittest.TestCase):
    def test_known_lookup_routes_to_rules(self) -> None:
        decision = gate_request(
            GateRequest(
                text="What is the path for config?",
                context={"known_paths": {"config": "config/app.yaml"}},
            )
        )

        self.assertEqual(decision.route, GateRoute.RULES)

    def test_repeated_request_routes_to_cache(self) -> None:
        decision = gate_request(
            GateRequest(
                text="Status check?",
                history=["Status check?", "Status check?"],
            )
        )

        self.assertEqual(decision.route, GateRoute.CACHE)

    def test_recent_failed_trajectory_escalates(self) -> None:
        decision = gate_request(
            GateRequest(
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
        )

        self.assertEqual(decision.route, GateRoute.EXPENSIVE_MODEL)
        self.assertEqual(decision.signals["failed_steps"], 2)

    def test_context_trajectory_accepts_plain_dicts(self) -> None:
        decision = gate_request(
            GateRequest(
                text="What is the path for config?",
                context={
                    "known_paths": {"config": "config/app.yaml"},
                    "trajectory": [
                        {
                            "domain": "terminal",
                            "action": "pwd",
                            "observation": "/repo",
                            "success": True,
                        }
                    ],
                },
            )
        )

        self.assertEqual(decision.route, GateRoute.RULES)
        self.assertEqual(decision.signals["domains"], ["terminal"])

    def test_coding_task_escalates(self) -> None:
        decision = gate_request(
            GateRequest(
                text="Write a function that returns the longest palindrome in a string.",
                context={"domain": "coding", "benchmark": "humaneval"},
            )
        )

        self.assertEqual(decision.route, GateRoute.EXPENSIVE_MODEL)
        self.assertTrue(decision.signals["coding_task"])

    def test_toy_eval_runs(self) -> None:
        result = run_toy_eval()

        self.assertGreaterEqual(result["total"], 1)
        self.assertEqual(result["correct"], result["total"])
        self.assertIn("deflection_rate", result)
        self.assertIn("estimated_cost_units", result)
        self.assertIn("avg_decision_ms", result)
        self.assertEqual(result["false_deflections"], 0)

    def test_coding_eval_runs_without_false_deflections(self) -> None:
        result = run_coding_eval()

        self.assertGreaterEqual(result["total"], 1)
        self.assertEqual(result["suite"], "coding")
        self.assertEqual(result["false_deflections"], 0)
        self.assertEqual(result["route_counts"][GateRoute.EXPENSIVE_MODEL.value], result["total"])

    def test_swebench_eval_runs_without_false_deflections(self) -> None:
        result = run_swebench_eval()

        self.assertGreaterEqual(result["total"], 1)
        self.assertEqual(result["suite"], "swebench")
        self.assertEqual(result["false_deflections"], 0)
        self.assertEqual(result["route_counts"][GateRoute.EXPENSIVE_MODEL.value], result["total"])


if __name__ == "__main__":
    unittest.main()
