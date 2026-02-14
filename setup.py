"""
Setup configuration for workflow_compiler package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README (prefer top-level README)
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "FlowCompile: Pareto-optimal agentic workflow compilation"

setup(
    name="flowcompile",
    version="0.1.0",
    author="Junyan Li",
    description="FlowCompile: Pareto-optimal agentic workflow compilation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["workflow_compiler", "workflow_compiler.*"]),
    include_package_data=True,
    package_data={"workflow_compiler.dsl": ["schema.json"]},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "pydantic>=2.0.0",
        "tqdm>=4.62.0",
        "aiofiles>=0.8.0",
        "tenacity>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "latency": [
            "vllm",
            "transformers>=4.20.0",
        ],
        "routers": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "scikit-learn>=1.0.0",
        ],
        "all": [
            "vllm",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "scikit-learn>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flowcompile=workflow_compiler.core.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
