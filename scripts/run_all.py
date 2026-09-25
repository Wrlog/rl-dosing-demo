"""Run the whole pipeline: train, evaluate, extra analyses, figures.

    python scripts/run_all.py            # full run, writes models/, results/, figures/
    python scripts/run_all.py --quick    # small version for CI, writes *_quick/ folders
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    extra = [a for a in sys.argv[1:] if a in ("--quick",)]
    t0 = time.time()
    for script in ("train.py", "evaluate.py", "analyses.py", "make_figures.py"):
        t = time.time()
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], check=True)
        print(f"== {script} took {time.time() - t:.0f} s", flush=True)
    print(f"pipeline finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
