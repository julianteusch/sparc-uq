# Release Checklist

## Initial Repository

- Standalone installable src-layout package, documented contracts and no datasets.
- Frozen-mean adapter and explicit custom callable interface.
- Feature precision, supplied covariance scaling, residual fallback and split CP.
- Synthetic examples, validation tests and optional research-core parity.
- CI definition, citation metadata and excluded environment/checkpoint artifacts.

## Before Claiming Full Paper Reproduction

- Public destination approved: https://github.com/julianteusch/sparc-uq.
- Apache-2.0 approved for the original repository code and documentation. Review
  upstream obligations separately for any future source/checkpoint imports.
- Audit actual staged files for secrets and private paths/data.
- Reproduce one locked paper configuration end to end, including actual mean,
  covariance checkpoint, preprocessing and coverage evaluation.
- Add that adapter and separately label the paper recipe vs generic defaults.
- Port structured operators before advertising large-trajectory efficiency.
- Validate against multiple Python/PyTorch versions and a clean installation.
- Document feature provenance, supported predictors and benchmark limitations.
- The project page and Hugging Face Space may link this initial core library with
  its scope clearly labeled; do not imply full paper reproduction before validation.

The initial library may be promoted with its scope explicitly labeled, without
claiming benchmark reproduction. See [publishing.md](publishing.md) for the
versioned GitHub release, wheel checks and PyPI Trusted Publishing setup.
Checkpoint releases and full paper reproduction remain separate work.
