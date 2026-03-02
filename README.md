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

## Next Steps

See our docs for further steps on how to create new workflows and optimize them for new tasks.


## Citation

If you find FlowCompile useful for your research or projects, please cite it as:

```bibtex

```
