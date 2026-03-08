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
    <a href="https://umass-embodied-agi.github.io/FlowCompile/" style='padding-left: 0.5rem;'>
      <img src="https://img.shields.io/badge/DOCS-ONLINE-0A9EDC?style=for-the-badge&logo=readthedocs&logoColor=white" alt='Docs'>
    </a>
    <a href="LICENSE" style='padding-left: 0.5rem;'><img src="https://img.shields.io/badge/LICENSE-MIT-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</p>

FlowCompile is an agentic LLM workflow compiler that computes the full accuracy–latency Pareto frontier, enabling principled configuration selection under diverse deployment preferences.

🚀 **Pareto-Optimal Compilation Toolchain**  
An end-to-end compilation pipeline that profiles sub-agents, searches a unified configuration space, and computes the full accuracy–latency Pareto frontier at compile time.

🧩 **Workflow DSL**  
Specify only the workflow structure using a PyTorch-like domain-specific language, while all profiling, optimization, and compilation are automatically handled by the FlowCompile engine.

📈 **Preference-Aware Deployment**  
Select workflow configurations at runtime via multiple strategies, including latency constraints, preference parameters, KNN routing, and more.

🛠️ **Unified CLI**  
An easy-to-use, end-to-end command-line interface covering profiling, compilation, and inference for every workflow.


## News

- **[2026-03]**: **FlowCompile** is officially released - easily implement, compile, optimize and run agentic workflow with our unified `flowcompile` CLI.


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


### Prepare Datasets

We provide support for optimization and evaluation on MATH-500, GSM8K, HotpotQA, and LiveCodeBench. The datasets for the first three benchmarks are already included under the `data/` directory. For LiveCodeBench, please run the following command to download and format the dataset:

```bash
python scripts/create_livecodebench_dataset.py
```

### CLI Commands

The `flowcompile` CLI is designed for ease of use through a unified configuration file.

Paper benchmark configs are provided in `configs/examples`:

- `configs/examples/flowcompile_hotpotqa.yaml`
- `configs/examples/flowcompile_gsm8k.yaml`
- `configs/examples/flowcompile_math500.yaml`
- `configs/examples/flowcompile_livecodebench.yaml`

1. Choose one benchmark YAML above and edit it for your experiment.
2. Run CLI commands with `--config`.
3. Use `--verbose`, `--quiet`, `--plain`, or `--no-banner` to control the new human-readable terminal output.

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

For custom OpenClaw Lobster workflows that already have profiling data, you can skip `prepare-data`.
Set these config keys and run `profile` directly:

- `workflow_type: openclaw_lobster`
- `openclaw_lobster_workflow_file: <path-to-lobster-yaml>`
- `profile_training_data: <path-to-training-json>`

Profiling judge policy for Outlook-style OpenClaw agents:

- `classify.category`: strict normalized exact match
- `summarize_each.summary`, `overview.overview_paragraph`, `ask_questions.question`, `draft_replies.draft_body`: semantic judge (`CORRECT`/`INCORRECT`) after JSON/required-field validation
- For `draft_replies.draft_body`, the judge checks semantic alignment and instruction adherence using the sample's `raw_llm_prompt`

Expected training sample fields for this path:

- top-level: `{"training_data": [...]}`
- per sample: `agent_name`, `raw_llm_prompt`, `processed_output` (JSON object string with required field), `raw_llm_output`

Optional end-to-end command:

```bash
flowcompile --config "$CONFIG" run-all
```

`run-all` executes `get-latency -> prepare-data -> profile -> predict -> test` in order.

Output mode examples:

```bash
flowcompile --verbose --config "$CONFIG" predict
flowcompile --plain --config "$CONFIG" run-all
flowcompile --quiet --config "$CONFIG" test
```


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

## Next Steps

To learn how to define custom workflows and optimize them for new tasks, please refer to our [documentation](https://umass-embodied-agi.github.io/FlowCompile/).


## Citation

If you find FlowCompile useful for your research or projects, please cite it as:

```bibtex

```
