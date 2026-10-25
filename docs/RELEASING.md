# Releasing

Releases are built and published by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
when a GitHub Release is published. PyPI never sees a password or token: the workflow authenticates with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).

> The publish job only runs when the repository variable `PUBLISH_TO_PYPI` is `true`, so a GitHub-only
> release (tag and notes, no upload) is the default. Set the variable after finishing the setup below.

## One-time setup (needs the PyPI account owner)

The project name `lumen-rag` was unclaimed on PyPI when this was written (checked 2026-10-18). The first
upload claims it, so register it before anyone else does.

1. Create a PyPI account with 2FA, if there is none: https://pypi.org/account/register/
2. Add a **pending publisher** (the project does not exist yet, so it is "pending"):
   https://pypi.org/manage/account/publishing/ → *Add a new pending publisher*
   - PyPI project name: `lumen-rag`
   - Owner: `WickTech`
   - Repository name: `lumen-rag`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repo: *Settings → Environments → New environment* named `pypi`. Add yourself as a
   required reviewer so every publish needs a manual approval click.
4. (Recommended) Do a dry run on TestPyPI first: add a second pending publisher on https://test.pypi.org
   and a `testpypi` environment, then point a copy of the publish step at
   `repository-url: https://test.pypi.org/legacy/`.

## Releasing a version

1. Update `__version__` in `lumen_rag/__init__.py` and the `CHANGELOG.md` heading (replace *unreleased*
   with the date).
2. Open a PR; make sure CI, the PR eval report and Security are green; merge.
3. Run the workflow's build and smoke test without publishing: *Actions → Release → Run workflow*
   (the `workflow_dispatch` path builds, runs `twine check`, and installs the wheel in a clean virtualenv).
4. Create the release: *Releases → Draft a new release*, tag `vX.Y.Z` on `master`, publish.
   The workflow refuses to run if the tag does not match `lumen_rag.__version__`.
5. Approve the `pypi` environment when GitHub asks. Verify:
   ```bash
   python -m venv /tmp/v && /tmp/v/bin/pip install lumen-rag && /tmp/v/bin/lumen --help
   ```
6. In `README.md`, change the install section to lead with `pip install lumen-rag`.

## Things that cannot be undone

- A published version number can never be reused, and a file can be *yanked* but not replaced. If a release
  is broken, publish a new patch version.
- Anything in the sdist or wheel is public. The sdist is limited by an include list in `pyproject.toml`
  (package, tests, eval data, scripts, docs); check `tar tzf dist/*.tar.gz` before a release.

## What the workflow checks

- Tag equals `lumen_rag.__version__`.
- `twine check` accepts the metadata and README.
- The wheel installs into a clean virtualenv away from the checkout, `lumen --help` runs, and a one-document
  ingest + query returns a citation.
