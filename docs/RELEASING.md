# Releasing

Releases are cut by pushing a git tag; CI builds the wheels and publishes to
PyPI. The version has a single source of truth — the Rust workspace version in
`Cargo.toml` — which maturin uses for the Python package (`pyproject.toml`
declares `dynamic = ["version"]`), and which `glmnet.__version__` reads back via
`importlib.metadata`.

## One-time setup

Publishing uses **PyPI trusted publishing** (OIDC), so there are no API tokens to
manage. Configure it once:

1. Create the project on PyPI (first upload can be done manually, or configure a
   "pending" trusted publisher before the first release).
2. On PyPI, add a trusted publisher for this repo:
   - Owner: `georgeberry`, Repository: `glmnet-rs`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In the GitHub repo settings, create an Environment named `pypi` (the
   `release.yml` `publish` job references it).

## Cutting a release

1. Bump the version in **`Cargo.toml`** (`[workspace.package] version`).
2. Move the notes from `## [Unreleased]` to a new `## [X.Y.Z]` section in
   `CHANGELOG.md` and update the compare/tag links at the bottom.
3. Commit on `main`:
   ```sh
   git commit -am "release: vX.Y.Z"
   git push origin main
   ```
4. Tag and push the tag:
   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

Pushing the tag triggers `release.yml`, which builds abi3 wheels for
Linux (x86_64, aarch64), macOS (x86_64, aarch64) and Windows (x64), plus an
sdist, then publishes them. The `publish` job only runs for tags whose base
branch is `main`.

## Notes

- **abi3 wheels**: the extension is built against CPython's stable ABI
  (`abi3-py39`), so a single wheel per OS/arch covers all CPython >= 3.9 — no
  per-Python-version build matrix.
- **Verify before tagging**: `cargo test -p glmnet-core --release`,
  `pytest tests/test_python.py`, and `maturin build --release` should all pass.
- To test the publish plumbing without touching real PyPI, point
  `gh-action-pypi-publish` at TestPyPI (`repository-url`) and add a matching
  trusted publisher there.
