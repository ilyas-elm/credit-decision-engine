"""Report the Kaggle image's environment, and whether this project installs on it.

Cheap to run and definitive, which beats guessing at the image contents and
discovering a conflict at the end of a 12 h feature build. It answers:

  1. which Python and which versions of the libraries this project uses are
     already present,
  2. whether optbinning and empulse -- not preinstalled -- pip-install cleanly
     on that Python, and what they drag in,
  3. whether installing them disturbs the preinstalled numpy/pandas/sklearn,
     which is the real risk of pinning versions that differ from the image.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "polars", "pyarrow", "numpy", "pandas", "sklearn",
    "lightgbm", "scipy", "matplotlib", "optbinning", "empulse",
]


def versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            out[name] = f"ABSENT ({type(exc).__name__})"
    return out


def main() -> None:
    report = {
        "python": sys.version,
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
        "before": versions(),
    }
    print("python:", report["python_version"], flush=True)
    print("BEFORE:", json.dumps(report["before"], indent=2), flush=True)

    # The two packages the project needs that the image does not carry.
    print("\n=== pip install optbinning empulse ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "optbinning", "empulse"],
        capture_output=True,
        text=True,
    )
    report["pip_returncode"] = proc.returncode
    report["pip_stdout"] = proc.stdout[-4000:]
    report["pip_stderr"] = proc.stderr[-4000:]
    print("return code:", proc.returncode, flush=True)
    if proc.stdout.strip():
        print(proc.stdout[-3000:], flush=True)
    if proc.stderr.strip():
        print("STDERR:", proc.stderr[-3000:], flush=True)

    # Re-import in a fresh interpreter: the current one has stale modules.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import json;"
         "out={};"
         f"names={PACKAGES!r};"
         "\nfor n in names:\n"
         "    try:\n"
         "        m=__import__(n); out[n]=getattr(m,'__version__','unknown')\n"
         "    except Exception as e:\n"
         "        out[n]=f'ABSENT ({type(e).__name__})'\n"
         "print(json.dumps(out))"],
        capture_output=True, text=True,
    )
    try:
        report["after"] = json.loads(probe.stdout.strip().splitlines()[-1])
    except Exception:
        report["after"] = {"error": probe.stdout[-2000:] + probe.stderr[-2000:]}
    print("\nAFTER:", json.dumps(report["after"], indent=2), flush=True)

    # Does the stack still work together once the two are installed?
    check = subprocess.run(
        [sys.executable, "-c",
         "import numpy as np, pandas as pd, polars as pl;"
         "from optbinning import BinningProcess, Scorecard;"
         "from empulse.metrics import empcs_score;"
         "import lightgbm as lgb;"
         "rng=np.random.default_rng(0); n=400;"
         "y=rng.binomial(1,0.2,n);"
         "X=pd.DataFrame({'a':rng.normal(0,1,n)+y});"
         "bp=BinningProcess(variable_names=['a'],"
         " binning_fit_params={'a':{'monotonic_trend':'auto_asc_desc'}});"
         "bp.fit(X,y);"
         "print('optbinning OK', float(bp.summary()['iv'].iloc[0]));"
         "print('empcs OK', float(empcs_score(y, rng.uniform(0,1,n))));"
         "print('lightgbm OK', lgb.__version__);"
         "print('polars OK', pl.__version__)"],
        capture_output=True, text=True,
    )
    report["smoke_returncode"] = check.returncode
    report["smoke_stdout"] = check.stdout
    report["smoke_stderr"] = check.stderr[-3000:]
    print("\n=== smoke test after install ===", flush=True)
    print("return code:", check.returncode, flush=True)
    print(check.stdout, flush=True)
    if check.returncode != 0:
        print("STDERR:", check.stderr[-3000:], flush=True)

    Path("/kaggle/working/env_report.json").write_text(json.dumps(report, indent=2))
    print("wrote env_report.json", flush=True)


if __name__ == "__main__":
    main()
