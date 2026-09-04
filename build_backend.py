"""Dependency-free PEP 517 backend for the pure-Python sidecar."""

from __future__ import annotations

import base64
import hashlib
import io
import pathlib
import zipfile

NAME = "vina_ad"
VERSION = "0.2.0"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
ROOT = pathlib.Path(__file__).parent


def _metadata() -> str:
    return (
        "Metadata-Version: 2.1\n"
        "Name: vina-ad\n"
        f"Version: {VERSION}\n"
        "Summary: ChainRules-compatible differentiable coordinate scoring sidecar for AutoDock Vina\n"
        "Requires-Python: >=3.10\n"
        "License: Apache-2.0\n"
    )


def _wheel_bytes() -> bytes:
    files: dict[str, bytes] = {}
    for path in sorted((ROOT / "vina_ad").glob("*.py")):
        files[f"vina_ad/{path.name}"] = path.read_bytes()
    files["vina_ad/requirements.md"] = (ROOT / "vina_ad/requirements.md").read_bytes()
    files[f"{DIST_INFO}/METADATA"] = _metadata().encode()
    files[f"{DIST_INFO}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: vina-ad-build-backend\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode()
    records: list[str] = []
    for name, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        records.append(f"{name},sha256={digest},{len(content)}")
    records.append(f"{DIST_INFO}/RECORD,,")
    files[f"{DIST_INFO}/RECORD"] = ("\n".join(records) + "\n").encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    path = pathlib.Path(metadata_directory) / DIST_INFO
    path.mkdir(parents=True, exist_ok=True)
    (path / "METADATA").write_text(_metadata(), encoding="utf-8")
    (path / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: vina-ad-build-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    return DIST_INFO


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    filename = f"{NAME}-{VERSION}-py3-none-any.whl"
    (pathlib.Path(wheel_directory) / filename).write_bytes(_wheel_bytes())
    return filename
