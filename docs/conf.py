from __future__ import annotations

import re
import sys
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_INIT = SRC_ROOT / "flowcompile" / "__init__.py"

sys.path.insert(0, str(SRC_ROOT))


def read_version() -> str:
    content = PACKAGE_INIT.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else "0.1.0"


project = "FlowCompile"
author = "FlowCompile contributors"
release = read_version()
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
nitpick_ignore = [
    ("py:class", "abc.ABC"),
    ("py:class", "pydantic.main.BaseModel"),
]

autodoc_mock_imports = [
    "aiofiles",
    "torch",
    "vllm",
    "litellm",
    "transformers",
    "datasets",
    "openai",
    "numpy",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "scipy",
    "scipy.stats",
    "sklearn",
    "tqdm",
    "tqdm.asyncio",
]

myst_enable_extensions = [
    "colon_fence",
]

html_theme = "sphinx_book_theme"
html_title = f"{project} Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "repository_url": "https://github.com/UMass-Embodied-AGI/FlowCompile",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
}
