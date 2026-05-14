# Installation

FlowCompile currently installs from source. The editable install exposes the
`flowcompile` CLI and the `flowcompile` Python package used by the paper
benchmark workflows.

## Requirements

- Python 3.11 is the primary supported target.
- A virtual environment is strongly recommended.
- Full experiment runs may use heavyweight dependencies such as `torch`, `vllm`,
  `litellm`, and local model-serving packages.
- Hosted or OpenAI-compatible model endpoints can be used through the model
  configuration file without running local vLLM workers.

## Install from Source

```bash
git clone https://github.com/UMass-Embodied-AGI/FlowCompile.git
cd FlowCompile

conda create -n flowcompile python=3.11
conda activate flowcompile

pip install -e .
```

Confirm the CLI imports:

```bash
flowcompile --help
```

The CLI has a flat config-driven interface with commands such as
`get-latency`, `prepare-data`, `profile`, `predict`, `test`, `runtime infer`,
and `experiments correlation`.

## Install Docs Dependencies Only

The documentation build deliberately avoids the project runtime dependency set.
For docs work, install the docs toolchain separately:

```bash
python -m pip install -r docs/requirements.txt
```

## Build the Documentation Site

From the repository root:

```bash
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

Or with the standard Sphinx make target:

```bash
make -C docs html
```

The HTML output is written to `docs/_build/html`.
