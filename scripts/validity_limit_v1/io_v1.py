"""Hash-verified, pickle-free loaders for frozen validity artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def load_json_verified(path: Path, expected: str) -> dict:
    verify_hash(path, expected)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_npz_verified(path: Path, expected: str) -> np.lib.npyio.NpzFile:
    verify_hash(path, expected)
    return np.load(path, allow_pickle=False)


def parse_metadata(npz: np.lib.npyio.NpzFile) -> dict:
    raw = npz["metadata_json"]
    if raw.shape != ():
        raise RuntimeError("metadata_json must be scalar")
    return json.loads(str(raw.item()))
