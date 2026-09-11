# Release Checklist

## Initial Repository

- Standalone installable src-layout package, documented contracts and no datasets.
- Frozen-mean adapter and explicit custom callable interface.
- Feature precision, supplied covariance scaling, residual fallback and split CP.
- Synthetic examples, validation tests and optional research-core parity.
- CI definition, citation metadata and excluded environment/checkpoint artifacts.

## Before a Public Code Announcement

- Confirm GitHub account/organization, repository name and public visibility.
- Agree on code license with authors and review upstream obligations for any future
  source/checkpoint imports. No license has been implicitly granted here.
- Audit actual staged files for secrets and private paths/data.
- Reproduce one locked paper configuration end to end, including actual mean,
  covariance checkpoint, preprocessing and coverage evaluation.
- Add that adapter and separately label the paper recipe vs generic defaults.
- Port structured operators before advertising large-trajectory efficiency.
- Validate against multiple Python/PyTorch versions and a clean installation.
- Document feature provenance, supported predictors and benchmark limitations.
- Only then link the code release from the project page and Hugging Face Space.

Publishing to GitHub, PyPI, Hugging Face or Overleaf is not part of creating this
local repository and has not been performed by the initial scaffold.
