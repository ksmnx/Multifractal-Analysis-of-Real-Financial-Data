import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "05_make_report_figures.py"),
                   run_name="__main__")
