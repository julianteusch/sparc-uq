# Initial Verification on bigbo

Date: 2026-09-11. Python 3.11.6, PyTorch 2.2.1.

## Environment

Repository: `/home/julianteusch/dev/sparc-uq`.

The repository has its own `.venv`. To avoid duplicating a large PyTorch/CUDA
installation, it was created using the existing `nam_for_dm` Python environment
with `--system-site-packages`. Development tools and this editable package were
installed into the new `.venv`; the existing environment was not modified.
Consequently this is not yet an isolated dependency-compatibility matrix test.
For a portable installation, follow README instructions with a fresh environment.

## Commands and Results

From the repository directory:

```bash
source .venv/bin/activate
SPARC_REFERENCE_ROOT=/home/julianteusch/dev/simlpe-pie-mp python -m pytest tests -q
python -m ruff check .
python examples/quickstart.py
python examples/structured_forecast.py
python -m build
```

- 31 tests passed, including the CUDA smoke test and numerical parity against
  the reference research functions. No paper datasets/checkpoints were evaluated.
- Ruff: all checks passed.
- Quickstart: unchanged mean; save/load equality; 0.9605 empirical mean marginal
  coverage on 2,000 synthetic test rows, for alpha=0.05.
- Structured example: unchanged mean; 0.9468333 empirical mean marginal coverage
  on 1,000 synthetic test trajectories with a specified covariance.
- Source distribution and wheel built successfully.

These coverage values are deterministic example observations, not evidence that
every finite test set attains nominal coverage or reproduces the paper's results.
The subsequent public GitHub Actions matrix also passed on Python 3.10 and 3.12:
29 tests passed per job; the CUDA and internal-reference tests were skipped as
expected on those CPU-only public runners. Lint, formatting, both examples and
package builds passed as well. The CPU-wheel configuration was verified in
[run 34572147776](https://github.com/julianteusch/sparc-uq/actions/runs/34572147776).
