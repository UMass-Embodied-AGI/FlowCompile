# Installation

FlowCompile currently installs from source. The documented path matches the repository setup in the project README.

## Requirements

- Python 3.11 is the primary supported target in `setup.py`.
- A virtual environment is strongly recommended.
- Full runtime installs may pull heavyweight dependencies such as `torch`, `vllm`, and `litellm`.

## Install from Source

```bash
git clone https://github.com/UMass-Embodied-AGI/FlowCompile.git
cd FlowCompile

conda create -n flowcompile python=3.11
conda activate flowcompile

pip install -e .
```

## Install Docs Dependencies Only

The documentation build deliberately avoids the project runtime dependency set. For docs work, install the docs toolchain separately:

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

