"""Exercise the consumer wheel, independently of the source/Hub import paths."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def test_installed_wheel_contains_the_formula_atlas(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(root / name, source / name)
    shutil.copytree(root / "torch-ext", source / "torch-ext",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
         "--no-build-isolation", "--wheel-dir", str(dist), str(source)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    consumer = tmp_path / "consumer"
    venv.EnvBuilder(with_pip=True).create(consumer)
    python = consumer / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
        check=True, capture_output=True, text=True, timeout=120,
    )
    consumer_code = (
        "import json; from importlib.resources import files; import szl_formulas as f; "
        "print(json.dumps({'atlas': f.load_formula_atlas(), "
        "'metadata': json.loads(files(f).joinpath('metadata.json').read_text(encoding='utf-8'))}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", consumer_code],
        cwd=tmp_path, check=False, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    package = root / "torch-ext" / "szl_formulas"
    assert actual["atlas"] == json.loads((package / "formula_atlas.v1.json").read_text(encoding="utf-8"))
    assert actual["metadata"] == json.loads((package / "metadata.json").read_text(encoding="utf-8"))
