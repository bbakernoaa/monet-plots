"""Release guards prevent tags and uploaded archives from misrepresenting versions."""

import importlib.metadata
import io
import json
import tarfile
import zipfile
from pathlib import Path

import monet_plots
import pytest

from scripts.check_release import check_release


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "monet-plots"\nversion = "0.2.0"\n'
    )
    (tmp_path / ".release-please-manifest.json").write_text(json.dumps({".": "0.2.0"}))
    return tmp_path


def archives(root, version="0.2.0"):
    dist = root / "dist"
    dist.mkdir()
    metadata = f"Name: monet-plots\nVersion: {version}\n".encode()
    with zipfile.ZipFile(dist / "monet_plots.whl", "w") as archive:
        archive.writestr("monet_plots.dist-info/METADATA", metadata)
    with tarfile.open(dist / "monet_plots.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("monet_plots/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
    return dist


def test_matching_release_and_archives(project):
    assert check_release(project, "v0.2.0", archives(project)) == "0.2.0"


@pytest.mark.parametrize("tag", ["v0.1.0", "0.2.0", "v0.2.0-rc.1"])
def test_wrong_tag_rejected(project, tag):
    with pytest.raises(ValueError, match="Tag"):
        check_release(project, tag)


def test_manifest_drift_rejected(project):
    (project / ".release-please-manifest.json").write_text('{".": "0.1.0"}')
    with pytest.raises(ValueError, match="versions differ"):
        check_release(project)


@pytest.mark.parametrize("version", ["01.2.0", "0.2", "0.2.0rc1"])
def test_invalid_stable_versions_rejected(project, version):
    path = project / "pyproject.toml"
    path.write_text(path.read_text().replace("0.2.0", version))
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        check_release(project)


def test_mismatched_archive_rejected(project):
    with pytest.raises(ValueError, match="Built distribution"):
        check_release(project, "v0.2.0", archives(project, "9.0.0"))


def test_extra_builds_rejected(project):
    dist = archives(project)
    (dist / "stale.whl").touch()
    with pytest.raises(ValueError, match="exactly one wheel"):
        check_release(project, dist_dir=dist)


def test_runtime_version_matches_installed_distribution():
    assert monet_plots.__version__ == importlib.metadata.version("monet-plots")


def test_repository_versions_agree():
    check_release(Path(__file__).resolve().parents[1])
