"""Open the local interface, or run a named SeePAT command."""

import importlib.util
import sys
from pathlib import Path

if __name__ == "__main__":
    from scripts.seepat_cli import main

    raise SystemExit(main(sys.argv[1:] or ["ui"]))

# Preserve package imports when this filename is found before src/seepat.
spec = importlib.util.spec_from_file_location(
    "seepat", Path(__file__).resolve().parent / "src/seepat/__init__.py"
)
package = importlib.util.module_from_spec(spec)
sys.modules[__name__] = package
spec.loader.exec_module(package)
