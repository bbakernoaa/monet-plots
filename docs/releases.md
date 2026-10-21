# Semantic versions and automatic releases

MONET Plots uses `MAJOR.MINOR.PATCH` versions. The package version lives in
`pyproject.toml`; Release Please updates it together with
`.release-please-manifest.json` and `CHANGELOG.md` in a release PR.
`monet_plots.__version__` reads installed package metadata, so it cannot drift
from the built distribution.

## Everyday workflow

Use Conventional Commit titles for PRs and **squash merge** them into `main`.
The PR title check enforces the syntax; configure squash merging to use the PR
title as the commit message. If merging without squash, the actual commits must
also follow the convention because Release Please reads commit history.

| Commit or squash-merged PR title | Version change |
| --- | --- |
| `fix(maps): correct raster extent` | Patch: 0.1.0 → 0.1.1 |
| `feat(maps): add spatial facets` | Minor: 0.1.0 → 0.2.0 |
| `feat!: change the plotting API` | Major: 0.1.0 → 1.0.0 |
| Any supported type with `BREAKING CHANGE:` in its commit body | Major |
| `docs: improve examples`, `test: cover facets`, `ci: update checks`, `chore: ...` | No release on their own |

Breaking changes increment the major version even before 1.0.0. Mark corrections
that change scientific interpretation as breaking when users must adapt their
analysis. This automation produces stable releases; prerelease channels are not
configured.

1. Merge feature/fix PRs into `main`.
2. After the complete CI matrix succeeds, Release Please opens or updates one
   release PR with the calculated version and release notes.
3. Review and merge that release PR. Its normal CI checks validate the version
   files and the plotting library before it can be released.
4. After CI succeeds on the merge commit, Release Please creates `vX.Y.Z` and a
   GitHub release. A reusable workflow builds a wheel and source archive from
   that tag, validates both archives' metadata, and attaches them to the release.

Do not manually bump version files or tag ordinary automated releases. The
highest applicable bump wins when a release contains several changes. The
bootstrap configuration starts from the current upstream commit and version
0.1.0; earlier repository history is excluded from the first generated changelog.

## One-time repository setup

A maintainer must configure these GitHub settings before automation can run:

- Add the Actions secret **`RELEASE_PLEASE_TOKEN`**: a fine-grained token for a
  bot/maintainer with access to this repository and **Contents: read/write** and
  **Pull requests: read/write**. Follow organization token approval policies.
  Release Please uses it to open/update the release PR and create the release.
- Allow GitHub Actions to create pull requests if repository/organization policy
  requires it. Enable squash merging with PR titles as squash commit messages.
- Make the CI matrix and `Conventional PR title / title` checks required on
  `main`. Require review of release PRs as appropriate for the project.

A separate token is necessary because PRs created with the default
`GITHUB_TOKEN` do not trigger the normal `pull_request` CI workflows. Distribution
uploads use the workflow's short-lived `GITHUB_TOKEN`, scoped to Contents: write;
no PyPI password or API key is involved. Non-release jobs have read-only Contents
permissions. Feature branches and fork PRs cannot run the release job.

This change only adds the configuration. It does not create credentials, alter
remote repository settings, create a tag, or publish a release immediately.

## Recovering an interrupted artifact upload

The tag and GitHub release are created before archives are uploaded. If packaging
or uploading fails, fix the underlying problem and run **Release distributions**
from the Actions tab with the existing tag (for example `v0.2.0`). The workflow
validates package/manifest/tag versions and replaces the two archive assets on
that release. GitHub Actions also retains the built archives as a workflow
artifact. Do not move or reuse a published tag for a different commit.

## PyPI publishing

GitHub releases are automated now. PyPI publishing is not enabled because PyPI
rejects direct URL dependencies, including the current Git-based `monet-stats`
requirement. Before adding PyPI automation:

1. Publish or identify an appropriate versioned `monet-stats` distribution and
   replace the Git requirement with that released dependency.
2. Register the project on PyPI and configure a GitHub Actions trusted publisher
   for this repository and a dedicated publishing workflow/environment.
3. Add an OIDC publishing job after the existing build and metadata checks.

Local checks can be run without any release credentials:

```bash
python scripts/check_release.py
python -m build
python scripts/check_release.py --tag v0.1.0 --dist-dir dist
```

Use the actual package version for the tag argument. There should be exactly one
wheel and one source archive in `dist/`; remove stale local builds before checking
a new version.
