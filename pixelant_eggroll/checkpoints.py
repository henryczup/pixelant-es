"""Checkpoint conversion and portable JAX parameter export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import DEFAULT_LAYOUT
from .models_jax import InverseGenerator


def _load_torch_state(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to convert `.pth` checkpoints.") from exc

    state = torch.load(Path(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a PyTorch state dict in {path}")
    return state


def _tensor(state: dict[str, Any], name: str) -> jnp.ndarray:
    value = state[name]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(np.asarray(value), dtype=jnp.float32)


def _torch_bn1d(state: dict[str, Any], prefix: str) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    params = {"gamma": _tensor(state, f"{prefix}.weight"), "beta": _tensor(state, f"{prefix}.bias")}
    frozen = {
        "mean": _tensor(state, f"{prefix}.running_mean"),
        "var": _tensor(state, f"{prefix}.running_var"),
        "eps": jnp.asarray(1e-5, dtype=jnp.float32),
    }
    return params, frozen


def inverse_from_torch_checkpoint(path: str | Path):
    """Convert `Net_inverse` `.pth` checkpoint to HyperscaleES/JAX pytrees."""

    state = _load_torch_state(path)
    fc_indices = sorted(
        int(key[2:].split(".")[0])
        for key in state
        if key.startswith("fc") and key.endswith(".weight")
    )
    if len(fc_indices) < 2:
        raise ValueError(f"Expected at least two fully connected layers in {path}")
    hidden_fc_indices = fc_indices[:-1]
    output_fc_index = fc_indices[-1]
    params = {}
    bn_frozen = {}
    for idx, fc_index in enumerate(hidden_fc_indices):
        params[f"fc{idx}"] = {"weight": _tensor(state, f"fc{fc_index}.weight"), "bias": _tensor(state, f"fc{fc_index}.bias")}
        params[f"bn{idx}"], bn_frozen[f"bn{idx}"] = _torch_bn1d(state, f"bn{fc_index}")
    params["out"] = {
        "weight": _tensor(state, f"fc{output_fc_index}.weight"),
        "bias": _tensor(state, f"fc{output_fc_index}.bias"),
    }

    hidden_dims = tuple(int(params[f"fc{idx}"]["weight"].shape[0]) for idx in range(len(hidden_fc_indices)))
    init = InverseGenerator.rand_init(jax.random.key(0), hidden_dims=hidden_dims)
    frozen = dict(init.frozen_params)
    for name, value in bn_frozen.items():
        frozen[name] = value
    frozen["layout"] = DEFAULT_LAYOUT.mask_shape
    frozen["latent_dim"] = 0
    frozen["spectrum_dim"] = 81
    return frozen, params, init.scan_map, init.es_map


def _torch_linear(state: dict[str, Any], prefix: str) -> dict[str, jnp.ndarray]:
    return {"weight": _tensor(state, f"{prefix}.weight"), "bias": _tensor(state, f"{prefix}.bias")}


def _torch_bn(state: dict[str, Any], prefix: str) -> dict[str, jnp.ndarray]:
    return {
        "gamma": _tensor(state, f"{prefix}.weight"),
        "beta": _tensor(state, f"{prefix}.bias"),
        "mean": _tensor(state, f"{prefix}.running_mean"),
        "var": _tensor(state, f"{prefix}.running_var"),
        "eps": jnp.asarray(1e-5, dtype=jnp.float32),
    }


def forward_surrogate_from_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    """Convert `Net_big` `.pth` checkpoint to JAX inference parameters."""

    state = _load_torch_state(path)
    params: dict[str, Any] = {}
    for idx in range(1, 17):
        params[f"conv{idx}"] = _torch_linear(state, f"conv{idx}")
        params[f"bn{idx}"] = _torch_bn(state, f"bn{idx}")
    for idx in (17, 18, 19):
        params[f"fc{idx}"] = _torch_linear(state, f"fc{idx}")
    for idx in (17, 18):
        params[f"bn{idx}"] = _torch_bn(state, f"bn{idx}")
    return params


def _flatten(prefix: str, tree: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(tree, dict):
        for key, value in tree.items():
            _flatten(f"{prefix}/{key}" if prefix else str(key), value, out)
    elif isinstance(tree, (tuple, list)):
        out[prefix] = np.asarray(tree)
    else:
        out[prefix] = np.asarray(tree)


def save_inverse_npz(path: str | Path, frozen_params: dict[str, Any], params: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Save the trained inverse generator as portable `.npz` plus JSON metadata."""

    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    _flatten("frozen", frozen_params, arrays)
    _flatten("params", params, arrays)
    np.savez_compressed(path, **arrays)
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _unflatten(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for name, value in arrays.items():
        node = root
        parts = name.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(value)
    return root


def _normalize_frozen_scalars(tree: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    tuple_keys = {"hidden_dims", "layout"}
    int_keys = {"latent_dim", "spectrum_dim", "input_dim", "output_dim", "channels", "height", "width"}
    for key, value in tree.items():
        if isinstance(value, dict):
            normalized[key] = _normalize_frozen_scalars(value)
        elif key in tuple_keys:
            normalized[key] = tuple(int(x) for x in np.asarray(value).reshape(-1))
        elif key in int_keys:
            normalized[key] = int(np.asarray(value).reshape(-1)[0])
        else:
            normalized[key] = value
    return normalized


def load_inverse_npz(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a portable generator checkpoint saved by `save_inverse_npz`.

    Returns `(frozen_params, params, metadata)`.  The checkpoint is intended for
    evaluation/export; scripts that need EGGROLL scan maps should recreate the
    matching generator config and then replace its `params` with the loaded tree.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        tree = _unflatten({name: data[name] for name in data.files})
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    frozen = _normalize_frozen_scalars(tree.get("frozen", {}))
    params = tree.get("params", {})
    return frozen, params, metadata
