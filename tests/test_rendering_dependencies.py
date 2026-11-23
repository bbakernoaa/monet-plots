"""Protect the core renderer from alternate plotting dependencies."""

import ast
from pathlib import Path

import monet_plots


def test_core_renderer_has_no_seaborn_imports():
    package = Path(monet_plots.__file__).parent
    importing_modules = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports = [node.module]
            else:
                continue
            if any(name.split(".")[0] == "seaborn" for name in imports):
                importing_modules.append(str(path.relative_to(package)))
    assert not importing_modules
