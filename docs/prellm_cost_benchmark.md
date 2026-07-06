# Exact pre-LLM gate cost benchmark

`tools/prellm_cost_benchmark.py` compares three strategies:

1. `baseline_all_advanced`: every case calls `openai/gpt-5.5-pro`.
2. `rules_pre_gate`: local deterministic `prellm-gate` runs first; only `expensive-model` cases call `openai/gpt-5.5-pro`.
3. `local_cheap_llm_pre_gate`: a local Ollama/OpenAI-compatible llama or Gemma endpoint acts as the gate; only `expensive-model` cases call `openai/gpt-5.5-pro`.

The score is a bounded proxy:

- route correctness against expected routes
- false deflections, especially SWE-bench Lite tasks routed away from `expensive-model`
- advanced-model token/cost usage
- gate token/cost usage

It is not a SWE-bench patch-resolution score. The harness does not check out repositories, apply patches, or run tests.

## Dry-run verification

Dry-run mode makes no provider calls. It estimates prompt tokens locally and uses fixed output-token assumptions so the harness can be tested without spending money.

```bash
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py --fixture-only
```

Use real SWE-bench Lite rows while still avoiding provider calls:

```bash
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py --swebench-length 3
```

## Live run

Live mode requires API access for the advanced model. By default, the harness uses the direct OpenAI API provider with `gpt-5.5-pro`.
ChatGPT web/desktop subscriptions are not API credentials; set `OPENAI_API_KEY` for programmable runs.
Token cost is estimated from the CLI rates unless provider usage returns a direct cost.

```bash
OPENAI_API_KEY=... \
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py \
  --execute \
  --swebench-length 3 \
  --output artifacts/prellm_cost_benchmark.live.json
```

OpenRouter remains available only as an explicit alternate provider:

```bash
OPENROUTER_API_KEY=... \
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py \
  --execute \
  --advanced-provider openrouter \
  --advanced-model openai/gpt-5.5-pro
```

Override rates if pricing changes:

```bash
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py \
  --execute \
  --advanced-input-cost-per-1m 30 \
  --advanced-output-cost-per-1m 180
```

## Local gate discovery

The local gate is discovered in this order:

- `--local-base-url`
- `PRELLM_GATE_LOCAL_URL`
- `LOCAL_LLM_BASE_URL`
- `OLLAMA_HOST`
- common local Ollama, OpenAI-compatible, and llama.cpp ports

It looks for model names containing `gemma` or `llama`, unless `--local-model` is provided.

Examples:

```bash
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py \
  --execute \
  --local-kind ollama \
  --local-base-url http://127.0.0.1:11434 \
  --local-model gemma3:4b
```

```bash
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py \
  --execute \
  --local-kind openai \
  --local-base-url http://127.0.0.1:1234/v1 \
  --local-model local-model-name
```

For llama.cpp servers that expose `/completion`:

```bash
PYTHONPATH=src python3 tools/prellm_cost_benchmark.py \
  --execute \
  --local-kind llamacpp \
  --local-base-url http://127.0.0.1:11435 \
  --local-model gemma
```

If no local endpoint is found, the local-gate strategy fails closed to `expensive-model` and marks gate tokens as dry-run estimates.
