# Benchmark Workflows

The paper evaluates FlowCompile on four public benchmarks across math,
multi-hop QA, and code reasoning. The repository exposes those workloads through
flat configs in `configs/examples/` and Python DSL workflows in
`workflow_compiler/workflows/`.

## Built-In Benchmarks

| Config | Workflow type | Dataset | Metric | Split files |
| --- | --- | --- | --- | --- |
| `flowcompile_gsm8k.yaml` | `gsm8k` | GSM8K | accuracy | `data/gsm8k_validate.jsonl`, `data/gsm8k_test.jsonl` |
| `flowcompile_math500.yaml` | `math` | MATH-500 | accuracy | `data/math500_validate.jsonl`, `data/math500_test.jsonl` |
| `flowcompile_hotpotqa.yaml` | `hotpotqa` | HotpotQA | F1 | `data/hotpotqa_validate.jsonl`, `data/hotpotqa_test.jsonl` |
| `flowcompile_livecodebench.yaml` | `livecodebench` | LiveCodeBench | Pass@1 | `data/livecodebench_validate.jsonl`, `data/livecodebench_test.jsonl` |

MATH-500, GSM8K, and HotpotQA split files are included in the repository.
LiveCodeBench data is generated separately:

```bash
python scripts/create_livecodebench_dataset.py
```

## Search Axes

The example configs search the same three axes discussed in the paper:

- `model`: which model alias each active sub-agent uses.
- `budget`: maximum generated reasoning tokens for that sub-agent call.
- `structure`: which inferred workflow structure variant is active.

The Qwen-3 model family used in the paper is represented by default
`latency_models`:

```text
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
Qwen/Qwen3-8B
Qwen/Qwen3-14B
```

To derive search aliases from this list, each model must have a unique
`hf_model_name` mapping in `configs/config.yaml`.

## Reasoning Budgets

The included configs use benchmark-specific discrete budget grids:

- GSM8K and HotpotQA use shorter QA-oriented grids up to `8000`.
- MATH-500 and LiveCodeBench use broader grids up to `16000`.

You can edit `search_budgets` in the experiment YAML or override it during
profiling and prediction.

## Workflow Structures

FlowCompile captures workflows from normal Python DSL code and infers structure
variants by pruning repeated or optional agent calls while preserving a valid
graph.

### Math and GSM8K

The shared math workflow contains:

- `programmer`
- `refine_solver`
- `detailed_solver`
- two `generate_solver` calls
- `sc_ensemble` when at least two branches feed aggregation
- `extract_answer` as a deterministic tool

The current implementation infers 17 valid structures. Low-latency configs can
use a single branch; high-accuracy configs can activate all four candidate
branches plus aggregation.

### HotpotQA

The HotpotQA workflow contains:

- one to three `answer_generate` calls
- optional `sc_ensemble` when multiple answers are active
- optional `format_answer`

The current implementation infers 6 valid structures. This matches the paper's
observation that HotpotQA often keeps a simple structure and shifts capacity
through model and budget allocation.

### LiveCodeBench

The LiveCodeBench workflow contains:

- one to three `code_generate` calls
- optional `sc_ensemble` when multiple programs are active
- `test` as a deterministic tool
- zero to three `reflection_test` repair attempts
- `select_final_solution` as a deterministic tool

The current implementation infers 12 valid structures. The bounded repair loop
is captured from `if test_out["test_passed"]: break` and handled by the
auto-backward proxy.

## Local Serving Setup

The paper latency results use a single H100 with vLLM. The repository includes
helper scripts for a local worker/judge setup:

```bash
bash scripts/setup_vllm/setup_vllm_models_worker.sh
bash scripts/setup_vllm/setup_vllm_models_judge.sh
```

Then edit `scripts/setup_vllm/litellm_config_1worker1judge.yaml` with real
hosts and launch LiteLLM:

```bash
litellm --config scripts/setup_vllm/litellm_config_1worker1judge.yaml --port 4000
```

Point local model entries in `configs/config.yaml` at the LiteLLM `base_url`
and proxy key.
