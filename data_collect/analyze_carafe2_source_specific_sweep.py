#!/usr/bin/env python3
"""Run the DIA-NN-v5-equivalent validation analysis for Carafe2 results."""
from __future__ import annotations

import sys
from pathlib import Path

from analyze_diann_source_specific_sweep import main as analyze


if __name__ == "__main__":
    # The shared analyzer owns argument parsing and all numerical/plot logic.
    # Copy a Carafe2-labelled guide only after a successful complete analysis.
    analyze()
    if "--study-root" not in sys.argv:
        raise SystemExit("--study-root is required")
    root = Path(sys.argv[sys.argv.index("--study-root") + 1]).resolve()
    guide = Path(__file__).with_name("diann_source_specific_plot_guide.md")
    target = root / "analysis" / "plots" / "README.md"
    target.write_text(guide.read_text(encoding="utf-8").replace("DIA-NN", "Carafe2"), encoding="utf-8")
    readme = root / "analysis" / "README.md"
    if readme.is_file():
        readme.write_text(readme.read_text(encoding="utf-8").replace("DIA-NN", "Carafe2"), encoding="utf-8")
