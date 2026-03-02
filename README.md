<p align="center">
  <h1 align="center">FlowCompile: Pareto-Optimal Agentic Workflow Compilation</h1>
  <p align="center">
    <a href="https://senfu.github.io/">Junyan Li</a>,
    <a href="https://williamd4112.github.io/">Zhang-Wei Hong</a>,
    <a href="https://maohaos2.github.io/Maohao/">Maohao Shen</a>,
    <a href="https://scholar.google.com/citations?user=_-5PSgQAAAAJ&hl=en">Yang Zhang</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='' style='padding-left: 0.5rem;'>
      <img src="https://img.shields.io/badge/DOCS-ONLINE-0A9EDC?style=for-the-badge&logo=readthedocs&logoColor=white" alt='Docs'>
    </a>
    <a href="LICENSE" style='padding-left: 0.5rem;'><img src="https://img.shields.io/badge/LICENSE-MIT-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</p>

FlowCompile is an agentic LLM workflow compiler that computes the full accuracy–latency Pareto frontier, enabling principled configuration selection under diverse deployment preferences.

🚀 **Pareto-Optimal Compilation**  
Compute the full accuracy–latency Pareto frontier of agentic workflows at compile time.

🧩 **Workflow DSL**  
Define and compose multi-stage LLM workflows using a PyTorch-like domain-specific language, enabling easy implementation and optimization.

📈 **Preference-Aware Deployment**  
Select workflow configurations at runtime via multiple strategies, including latency constraints, preference parameters, KNN routing, and more.

🛠️ **Unified CLI**  
End-to-end command-line interface for profiling, compilation, and inference.


## News

- **[2026-03]**: **FlowCompile** is officially released - easily implement, compile, optimize and run agentic workflow with our unified `flowcompile` CLI.


## Contents

- [Install](#install)
- [Configure Models](#configure-models)
- [Quickstart (YAML Config CLI)](#quickstart-yaml-config-cli)
- [Add a New Benchmark](#add-a-new-benchmark)
- [Add a New Workflow](#add-a-new-workflow)
- [Experiment](#experiment)
- [Runtime](#runtime)

## Get Started

### Installation

#### Install from source

Clone the repository and install in editable mode using a virtual environment (e.g., with conda):

```bash
# Clone the repository
git clone https://github.com/UMass-Embodied-AGI/FlowCompile.git
cd FlowCompile

# Create and activate virtual environment
conda create -n flowcompile python=3.11
conda activate flowcompile

# Install the package
pip install -e .
```

### Configure Models & API Keys

Create your model config and set API keys/env vars required by your backend.

```bash
cp configs/config.example.yaml configs/config.yaml
```

To reproduce the results in our paper with local endpoints, set up `vllm` worker
and judge servers, then use LiteLLM as a single local API endpoint for all
models:

1. Launch worker `vllm` servers:

```bash
bash scripts/setup_vllm/setup_vllm_models_worker.sh
```

2. Launch judge `vllm` servers (on a different machine from worker, or update
   ports/device mapping to avoid conflicts):

```bash
bash scripts/setup_vllm/setup_vllm_models_judge.sh
```

3. Update `scripts/setup_vllm/litellm_config_1worker1judge.yaml` with real
   `WORKER_IP` and `JUDGE_IP`.

4. Start LiteLLM proxy for unified local OpenAI-compatible API access:

```bash
litellm --config scripts/setup_vllm/litellm_config_1worker1judge.yaml --port 4000
```

5. In `configs/config.yaml`, point OpenAI-style models to the LiteLLM endpoint
   (`base_url: "http://127.0.0.1:4000"`) and use the LiteLLM `master_key` as
   `api_key`.

### CLI Commands

The `flowcompile` CLI is designed for ease of use through a unified configuration file.

Paper benchmark configs are provided in `configs/examples`:

- `configs/examples/flowcompile_hotpotqa.yaml`
- `configs/examples/flowcompile_gsm8k.yaml`
- `configs/examples/flowcompile_math500.yaml`
- `configs/examples/flowcompile_livecodebench.yaml`

1. Choose one benchmark YAML above and edit it for your experiment.
2. Run CLI commands with `--config`.

```bash
CONFIG=configs/examples/flowcompile_math500.yaml

# 0) Benchmark latency
flowcompile --config "$CONFIG" get-latency

# 1) Prepare profiling data (ground-truth + agent dataset)
flowcompile --config "$CONFIG" prepare-data

# 2) Profile agent performance (uses search_budgets)
flowcompile --config "$CONFIG" profile

# 3) Compile Pareto configs
flowcompile --config "$CONFIG" predict

# 4) Evaluate compiled Pareto configs
flowcompile --config "$CONFIG" test
```

Optional end-to-end command:

```bash
flowcompile --config "$CONFIG" run-all
```

`run-all` executes `get-latency -> prepare-data -> profile -> predict -> test` in order.


### Analysis

We support correlation analysis through our unified CLI. To check the correlation between predicted and actual workflow accuracy/latency:

```bash
flowcompile --config "$CONFIG" experiments correlation
```

### Runtime

We support running workflows using compiled configurations directly through our unified CLI. The primary usage is as follows:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget 0.5
```

Named runtime preference budgets (`low`, `medium`, `high` and `xhigh`) are also supported. Example:

```bash
flowcompile --config "$CONFIG" runtime infer \
  --query "Solve 1+1" \
  --strategy preference \
  --budget high
```

Single-query output is now human-readable and includes the selected config, sub-agent settings, final workflow output, and measured wall-clock runtime. Example:

```text
Used Config
  Config ID: cfg_0019
  Structure ID: s__programmer-c0__refine_solver-c0__detailed_solver-c0__generate_solver-c2__sc_ensemble-c1
  Sub-agents:
    generate_solver: setting=qwen3-1.7b_budget_10, model=qwen3-1.7b, budget=10
    sc_ensemble: setting=qwen3-8b_budget_10, model=qwen3-8b, budget=10

Workflow Output
  2

Actual Runtime
  4.237s

Metadata
  Query ID: q1
  Output Dir: results/math500/runtime/outputs/q1
```


## Add a New Benchmark

1. Create a new benchmark module:
   - `workflow_compiler/benchmarks/<name>.py`
2. Implement a benchmark class:
   - Subclass `BaseBenchmark` from `workflow_compiler/benchmarks/benchmark.py`
   - Add `@register_benchmark()` from `workflow_compiler/benchmarks/registry.py`
   - Define class metadata:
     - `BENCHMARK_NAME`
     - `ALIASES`
     - `WORKFLOW_TYPE`
     - `METRIC_NAME`
     - `DEFAULT_SPLIT_PATHS` (`{"validate": "...", "test": "..."}`)
   - Implement required methods:
     - `evaluate_problem(...)`
     - `calculate_score(...)`
     - `get_result_columns(...)`
3. Add your dataset files under `data/` and point `DEFAULT_SPLIT_PATHS` to them.
4. Point your YAML config (`configs/examples/*.yaml`) at your benchmark and files:
   - `dataset: <benchmark alias or canonical name>`
   - `workflow_type: <workflow type used by the benchmark>`
   - `validate_file` / `test_file`
5. Verify auto-registration (no central benchmark list edit is needed):

```bash
python - <<'PY'
from workflow_compiler.benchmarks import list_benchmarks
for row in list_benchmarks(detailed=True):
    print(row["name"], row.get("aliases", []), row.get("workflow_type"), row.get("metric_name"))
PY
```

Reference:
- `workflow_compiler/benchmarks/ADDING_BENCHMARK.md`

## Add a New Workflow

1. Create a workflow folder under `workflow_compiler/workflows/<workflow_name>/`.
2. Start from the template:
   - `workflow_compiler/workflows/template/workflow.py`
   - Implement a `WorkflowModule` subclass with:
     - `workflow_type = "<workflow_name>"`
     - `forward(self, query)` returning:
       - `final_answer`
       - `full_solution`
       - `final_solution`
3. Register it in `workflow_compiler/workflows/dsl_registry.py` in `get_workflow_module(...)`.
4. If you add new agent names, register factories in `workflow_compiler/dsl/registry.py` (`get_agent_factory`).
5. If you add new tool nodes, register implementations in `workflow_compiler/dsl/registry.py` (`TOOL_REGISTRY`).
6. If this is a brand-new workflow type, also update workflow-type handling in:
   - `workflow_compiler/core/cli.py` (`_validate_flat_config`)
   - `workflow_compiler/dsl/runtime.py` (`_preprocess_query`, trace builder logic)
   - `workflow_compiler/runtime/engine.py` (runtime LLM config mapping)
   - `workflow_compiler/compiler/validation.py` (`_build_llm_configs_for_workflow`)
7. Create a new config file in `configs/examples/` and run the pipeline:

```bash
CONFIG=configs/examples/flowcompile_<workflow_name>.yaml
flowcompile --config "$CONFIG" run-all
```

Quick registry sanity check:

```bash
python - <<'PY'
from workflow_compiler.workflows.dsl_registry import get_workflow_module
wf = get_workflow_module("<workflow_name>")
print(wf.get_full_structure()["structure_id"])
PY
```

Reference:
- `workflow_compiler/workflows/ADDING_WORKFLOW.md`
