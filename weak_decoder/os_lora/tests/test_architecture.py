"""约束 OS-LoRa 系统、实验支持代码和实验入口之间的依赖方向。"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
import unittest


OS_LORA_ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> tuple[tuple[str, int], ...]:
    """返回源码中的导入模块及其行号。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = "." * node.level + (node.module or "")
            else:
                module = node.module or ""
            imports.append((module, node.lineno))
    return tuple(imports)


def _resolve_import(module: str, package: str) -> str:
    """把相对导入解析为便于检查的绝对模块名。"""

    level = len(module) - len(module.lstrip("."))
    if level == 0:
        return module
    target = module.lstrip(".")
    package_parts = package.split(".")
    base_parts = package_parts[: len(package_parts) - level + 1]
    if target:
        base_parts.extend(target.split("."))
    return ".".join(base_parts)


class ArchitectureTests(unittest.TestCase):
    """防止后续修改重新引入跨层或实验入口之间的依赖。"""

    def test_experiment_entries_do_not_import_each_other(self) -> None:
        """任一实验入口都不能导入另一个实验入口。"""

        experiment_dir = OS_LORA_ROOT / "experiments"
        violations: list[str] = []
        for path in sorted(experiment_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            for module, line in _imports(path):
                resolved = _resolve_import(module, "weak_decoder.os_lora.experiments")
                if resolved.startswith("weak_decoder.os_lora.experiments"):
                    violations.append(f"{path.name}:{line} -> {resolved}")
        self.assertEqual([], violations, "实验入口存在相互依赖：\n" + "\n".join(violations))

    def test_system_has_no_experiment_dependencies(self) -> None:
        """在线系统实现不能依赖实验入口或离线支持代码。"""

        violations: list[str] = []
        for path in sorted((OS_LORA_ROOT / "system").glob("*.py")):
            for module, line in _imports(path):
                resolved = _resolve_import(module, "weak_decoder.os_lora.system")
                if resolved.startswith("weak_decoder.os_lora.experiments") or resolved.startswith(
                    "weak_decoder.os_lora.experiment_support"
                ):
                    violations.append(f"{path.name}:{line} -> {resolved}")
        self.assertEqual([], violations, "system 出现反向依赖：\n" + "\n".join(violations))

    def test_experiment_support_does_not_import_entries(self) -> None:
        """实验共享支持代码不能反向依赖某个具体实验入口。"""

        violations: list[str] = []
        for path in sorted((OS_LORA_ROOT / "experiment_support").glob("*.py")):
            for module, line in _imports(path):
                resolved = _resolve_import(module, "weak_decoder.os_lora.experiment_support")
                if resolved.startswith("weak_decoder.os_lora.experiments"):
                    violations.append(f"{path.name}:{line} -> {resolved}")
        self.assertEqual([], violations, "实验支持代码依赖了入口：\n" + "\n".join(violations))

    def test_no_duplicate_system_modules_at_package_root(self) -> None:
        """系统模块只保留一份实现，不在 os_lora 根目录设置同名副本。"""

        duplicates = [
            path.name
            for path in sorted((OS_LORA_ROOT / "system").glob("*.py"))
            if path.name != "__init__.py" and (OS_LORA_ROOT / path.name).exists()
        ]
        self.assertEqual([], duplicates)

    def test_every_experiment_entry_imports_independently(self) -> None:
        """逐个导入现存实验入口，及时发现移动后遗漏的依赖。"""

        package = importlib.import_module("weak_decoder.os_lora.experiments")
        names = sorted(module.name for module in pkgutil.iter_modules(package.__path__))
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(f"{package.__name__}.{name}")


if __name__ == "__main__":
    unittest.main()
