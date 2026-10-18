"""Bootstrap for the offline checks: Forge Neo's real sources, no GPU, no checkpoint, no webui.

Run a check with Forge Neo's own interpreter, from anywhere:

    venv/Scripts/python.exe extensions/neo-krea2-negpip/tests/check_moss.py

`FORGE` is this repository's grandparent when it lives in Forge's `extensions/`, else the
`FORGE_NEO` environment variable.  Importing this module runs `initialize.imports()`, so
it has to be imported before anything from `backend` or `modules`.
"""

import os
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]

if (EXT.parent.name == "extensions") and (EXT.parents[1] / "modules" / "initialize.py").exists():
    FORGE = EXT.parents[1]
else:
    FORGE = Path(os.environ.get("FORGE_NEO", ""))

if not FORGE.name or not (FORGE / "modules" / "initialize.py").exists():
    raise SystemExit(f"Forge Neo not found at {FORGE or '<unset>'}; set FORGE_NEO to its folder")

os.chdir(FORGE)
sys.path.insert(0, str(FORGE))
sys.path.insert(0, str(FORGE / "modules_forge" / "packages"))
sys.path.insert(0, str(EXT))
sys.argv = [sys.argv[0], "--skip-python-version-check", "--skip-torch-cuda-test", "--skip-install", "--skip-prepare-environment", "--ui-debug-mode"]

from modules import initialize  # noqa: E402

initialize.imports()

from backend import memory_management  # noqa: E402

#   no checkpoint is ever loaded; the text hook calls this before encoding
memory_management.load_model_gpu = lambda *a, **k: None

failures = 0


def check(name: str, ok: bool):
    global failures
    print(("PASS " if ok else "FAIL ") + name)
    failures += 0 if ok else 1


def finish():
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


def load_script():
    """The extension's script, imported by path the way Forge does, with `basedir` resolving."""
    import importlib.util

    from modules import scripts as forge_scripts

    forge_scripts.current_basedir = str(EXT)
    spec = importlib.util.spec_from_file_location("neo_krea2_negpip_script", EXT / "scripts" / "neo_krea2_negpip.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Prompt(list):
    is_negative_prompt = False
