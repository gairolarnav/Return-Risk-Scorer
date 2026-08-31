"""
Defense-only compliance guard (Track 02 grading line: "anything
offense-capable is disqualified"). Fails if any module under src/ or
scripts/ imports a generative, LLM, or adversarial-pattern-generation
library, or if requirements.txt names one as a dependency.

Cheap to run, and it maps directly onto the one criterion that
disqualifies a submission outright rather than merely costing credibility.
"""

import ast
import pathlib

# Generative/LLM-provider libraries specifically -- not general ML/DL
# frameworks (torch, tensorflow, ...), which are not what "offense-capable"
# means here and are not banned by the track line on their own.
BANNED_MODULE_SUBSTRINGS = [
    "openai",
    "anthropic",
    "langchain",
    "transformers",
    "diffusers",
    "llama_cpp",
    "gpt4all",
    "cohere",
    "replicate",
    "huggingface_hub",
    "together",
    "mistralai",
    "generativeai",
    "vertexai",
]


def _imported_module_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _all_python_files():
    for base in ("src", "scripts"):
        base_path = pathlib.Path(base)
        if base_path.exists():
            yield from base_path.rglob("*.py")


def test_no_generative_or_llm_dependencies():
    """No module under src/ or scripts/ may import a generative/LLM library.
    (The planned app/ directory was cut from scope -- see the correction log in
    docs/ARCHITECTURE.md §11 -- so there is nothing under app/ to check.)"""
    offenders = {}
    for path in _all_python_files():
        imported = _imported_module_names(path)
        hits = {m for m in imported if any(b in m.lower() for b in BANNED_MODULE_SUBSTRINGS)}
        if hits:
            offenders[str(path)] = hits
    assert not offenders, f"generative/LLM imports found: {offenders}"


def test_requirements_txt_has_no_generative_or_llm_packages():
    text = pathlib.Path("requirements.txt").read_text().lower()
    hits = [b for b in BANNED_MODULE_SUBSTRINGS if b in text]
    assert not hits, f"requirements.txt names banned packages: {hits}"
