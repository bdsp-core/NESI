"""Run a script unchanged, saving any figure it would only have displayed with plt.show()."""
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

script, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
count = [0]


def show(*args, **kwargs):
    for num in plt.get_fignums():
        count[0] += 1
        plt.figure(num).savefig(outdir / f"{script.stem}_{count[0]}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


plt.show = show
sys.argv = [script.name]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
show()
