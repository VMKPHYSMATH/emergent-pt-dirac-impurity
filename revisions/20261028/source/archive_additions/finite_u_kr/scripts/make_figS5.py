"""Compatibility entry point: regenerate main Fig. 7 and SM Fig. S5."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("make_filling_figures.py")), run_name="__main__")
