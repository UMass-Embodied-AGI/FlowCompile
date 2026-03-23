from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


readme_file = ROOT / "README.md"
requirements_file = ROOT / "requirements.txt"

if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "FlowCompile: Pareto-optimal agentic workflow compilation"

setup(
    name="flowcompile",
    version="0.1.0",
    author="Junyan Li",
    description="FlowCompile: Pareto-optimal agentic workflow compilation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=(
        find_packages(include=["workflow_compiler", "workflow_compiler.*"])
        + find_packages(where="3rdparty/flashflow", include=["flashflow", "flashflow.*"])
    ),
    package_dir={
        "flashflow": "3rdparty/flashflow/flashflow",
    },
    include_package_data=True,
    package_data={
        "workflow_compiler.dsl": ["schema.json"],
        "workflow_compiler.benchmarks": ["long_text.txt"],
    },
    python_requires=">=3.8",
    install_requires=read_requirements(requirements_file),
    entry_points={
        "console_scripts": [
            "flowcompile=workflow_compiler.core.cli:main",
            "flashflow=flashflow.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
    ],
)
