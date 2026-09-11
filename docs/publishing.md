# Publishing SPARC-UQ

Release 0.1.0 contains the reusable core, executed synthetic notebook and optional
visualization helpers, not the full paper benchmark implementation. Public code
and Apache-2.0 licensing are approved. PyPI account-side setup is still required;
do not advertise `pip install sparc-uq` as available until an actual upload has
succeeded. The versioned GitHub wheel can be installed directly after its release.

## Release Checks

1. Update `pyproject.toml`, `CITATION.cff` and `CHANGELOG.md` together.
2. Review the staged files for private paths, secrets, datasets and checkpoints.
3. Run Ruff, the test suite, examples and the notebook in a clean environment.
4. Build wheel and sdist, run `python -m twine check --strict dist/*`, and test
   the installed wheel rather than an editable checkout.
5. Check that README links resolve outside GitHub and that install commands use
   the actual distribution name `sparc-uq` (the import is `sparc`).
6. Push an annotated `vX.Y.Z` tag only after the normal Python 3.10/3.12 CI passes.
7. Run the Release workflow on that tag. It re-runs tests against the wheel,
   executes the notebook and publishes wheel, sdist and SHA256SUMS on GitHub.

```bash
gh workflow run release.yml --ref v0.1.0 -f publish_pypi=false
```

The workflow is manual and tag-only. Publishing permissions are limited to their
respective jobs; the build job has no OIDC or repository write permission.
Existing GitHub release assets are never overwritten by a re-run.

## One-Time PyPI Setup

Sign into your own PyPI account (with its required account security configured).
At https://pypi.org/manage/account/publishing/, add a **pending GitHub publisher**:

| Field | Value |
| --- | --- |
| PyPI project name | `sparc-uq` |
| Owner | `julianteusch` |
| Repository | `sparc-uq` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

This does not reserve the package name. No API token needs to be sent, committed
or copied to bigbo. See the official
[pending-publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

In the GitHub repository, create the `pypi` environment, restrict deployments to
release tags (`v*`), and configure a required reviewer before enabling uploads.
Then run:

```bash
gh workflow run release.yml --ref v0.1.0 -f publish_pypi=true
```

Approve the environment deployment, verify the workflow and PyPI project, then
test installation from PyPI in a fresh environment. Only afterwards update the
README, project page, Hugging Face Space and announcement to show:

```bash
python -m pip install 'sparc-uq[visualization]==0.1.0'
```

PyPI versions are immutable. Do not re-use an uploaded version for changed code.
If publishing fails, inspect the cause before retrying. Do not use
`skip-existing` to hide unexpected distribution conflicts.

## Scientific Scope

The initial library can be promoted without implementing the complete paper
recipe, provided its scope is explicit. The synthetic notebook is an integration
demonstration, not evidence of matching the reported benchmark results. Signed
feature-leverage terms are not causal importance or SHAP values. Coverage is
subject to exchangeability and the selected prediction unit.
