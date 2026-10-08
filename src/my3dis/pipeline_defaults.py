"""Centralised default paths and constants for the My3DIS pipeline."""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(env_var: str, fallback: Path) -> Path:
    """Return a Path honouring environment overrides."""

    override = os.environ.get(env_var)
    if not override:
        return fallback

    override_path = Path(override).expanduser()
    try:
        return override_path.resolve()
    except FileNotFoundError:
        # If the target does not exist yet we still propagate the intended absolute path.
        return override_path.absolute()


# Identify the project root so relative fallbacks remain stable even when launched elsewhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Semantic-SAM assets -----------------------------------------------------
_DEFAULT_SEMANTIC_SAM_ROOT = _PROJECT_ROOT / "third_party" / "Semantic-SAM"
DEFAULT_SEMANTIC_SAM_ROOT = _path_from_env("MY3DIS_SEMANTIC_SAM_ROOT", _DEFAULT_SEMANTIC_SAM_ROOT)
DEFAULT_SEMANTIC_SAM_CKPT = _path_from_env(
    "MY3DIS_SEMANTIC_SAM_CKPT",
    DEFAULT_SEMANTIC_SAM_ROOT / "ckpts" / "swinl_only_sam_many2many.pth",
)

# SAM2 assets -------------------------------------------------------------
_DEFAULT_SAM2_ROOT = _PROJECT_ROOT / "third_party" / "Grounded-SAM-2"
DEFAULT_SAM2_ROOT = _path_from_env("MY3DIS_SAM2_ROOT", _DEFAULT_SAM2_ROOT)
DEFAULT_SAM2_CFG = _path_from_env(
    "MY3DIS_SAM2_CFG",
    DEFAULT_SAM2_ROOT / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
)
DEFAULT_SAM2_CKPT = _path_from_env(
    "MY3DIS_SAM2_CKPT",
    DEFAULT_SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
)

# Example dataset + output layout ----------------------------------------
DEFAULT_DATA_PATH = _path_from_env(
    "MY3DIS_DATA_PATH",
    _PROJECT_ROOT / "data" / "multiscan" / "scene_00065_00" / "outputs" / "color",
)
DEFAULT_OUTPUT_ROOT = _path_from_env(
    "MY3DIS_OUTPUT_ROOT",
    _PROJECT_ROOT / "outputs" / "scene_00065_00",
)


def expand_default(path: Path) -> str:
    """Return an absolute string path for defaults, preserving user overrides."""

    try:
        return str(path.expanduser().resolve())
    except FileNotFoundError:
        return str(path.expanduser().absolute())
