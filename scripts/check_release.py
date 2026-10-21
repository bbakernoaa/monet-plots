"""Validate release versions before building or uploading distribution artifacts."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

STABLE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def check_release(root: Path, tag: str | None = None, dist_dir: Path | None = None):
    """Require manifest, package metadata, release tag and archives to agree."""
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if not STABLE_VERSION.fullmatch(version):
        raise ValueError(
            f"Expected a stable MAJOR.MINOR.PATCH version, got {version!r}"
        )
    manifest = json.loads((root / ".release-please-manifest.json").read_text())
    if manifest.get(".") != version:
        raise ValueError("Release Please manifest and pyproject.toml versions differ")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match package version v{version}")
    if dist_dir is not None:
        wheels = list(dist_dir.glob("*.whl"))
        sources = list(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sources) != 1:
            raise ValueError("Expected exactly one wheel and one source archive")
        with zipfile.ZipFile(wheels[0]) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(
                    "Wheel must contain exactly one distribution metadata file"
                )
            metadata = [archive.read(names[0])]
        with tarfile.open(sources[0], "r:gz") as archive:
            members = [
                m
                for m in archive.getmembers()
                if len(Path(m.name).parts) == 2 and m.name.endswith("/PKG-INFO")
            ]
            if len(members) != 1:
                raise ValueError("Source archive must contain top-level PKG-INFO")
            metadata.append(archive.extractfile(members[0]).read())
        for content in metadata:
            headers = BytesParser().parsebytes(content)
            if headers["Name"] != project["name"] or headers["Version"] != version:
                raise ValueError(
                    "Built distribution name/version differs from release metadata"
                )
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--dist-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(f"Validated release {check_release(root, args.tag, args.dist_dir)}")


if __name__ == "__main__":
    main()
