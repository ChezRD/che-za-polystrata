"""Resolve a parquet the audio checks read.

A plain path is used as given. ``hf://datasets/<org>/<name>/<file>`` is the file
in that Hugging Face dataset, downloaded into the standard Hub cache
(``${HF_HOME:-$HOME/.cache/huggingface}/hub``).
"""
from __future__ import annotations

from pathlib import Path


def resolve_data(spec: str) -> Path:
    if not spec.startswith("hf://"):
        return Path(spec)
    rest = spec[len("hf://"):]
    if rest.startswith("datasets/"):
        rest = rest[len("datasets/"):]
    org, sep, tail = rest.partition("/")
    name, sep2, filename = tail.partition("/")
    if not sep or not sep2 or not org or not name or not filename:
        raise SystemExit(f"{spec}: expected hf://datasets/<org>/<name>/<file>")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=f"{org}/{name}", filename=filename, repo_type="dataset"))
