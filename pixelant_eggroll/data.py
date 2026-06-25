"""Dataset loading for existing MATLAB `.mat` antenna data files."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io


SPECTRUM_KEYS = ("YTrain", "specd", "spec", "ss", "ydata")
DESIGN_KEYS = ("XTrain1", "designd", "Test_patches", "xdata", "designs")


def load_spectra_mat(path: str | Path, key: str | None = None) -> jnp.ndarray:
    mat = scipy.io.loadmat(Path(path))
    if key is None:
        key = next((candidate for candidate in SPECTRUM_KEYS if candidate in mat), None)
    if key is None or key not in mat:
        raise KeyError(f"Could not find a spectrum matrix in {path}; tried {SPECTRUM_KEYS}")
    data = np.asarray(mat[key], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D spectrum matrix, got shape {data.shape}")
    if data.shape[1] == 81:
        spectra = data
    elif data.shape[0] == 81:
        spectra = data.T
    else:
        raise ValueError(f"Expected one spectrum dimension to be 81, got {data.shape}")
    return jnp.asarray(spectra, dtype=jnp.float32)


def load_paired_antenna_mat(
    path: str | Path,
    spectrum_key: str | None = None,
    design_key: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load paired spectra and binary 12x12 designs from existing `.mat` datasets.

    Returns:
        spectra: `[N, 81]` float32
        designs: `[N, 1, 12, 12]` float32 binary masks
    """

    mat_path = Path(path)
    mat = scipy.io.loadmat(mat_path)
    if spectrum_key is None:
        spectrum_key = next((candidate for candidate in SPECTRUM_KEYS if candidate in mat), None)
    if design_key is None:
        design_key = next((candidate for candidate in DESIGN_KEYS if candidate in mat), None)
    if spectrum_key is None or spectrum_key not in mat:
        raise KeyError(f"Could not find a spectrum matrix in {path}; tried {SPECTRUM_KEYS}")
    if design_key is None or design_key not in mat:
        raise KeyError(f"Could not find a design matrix in {path}; tried {DESIGN_KEYS}")

    spectra = _normalize_spectra(np.asarray(mat[spectrum_key], dtype=np.float32))
    designs = _normalize_designs(np.asarray(mat[design_key]))
    if spectra.shape[0] != designs.shape[0]:
        count = min(spectra.shape[0], designs.shape[0])
        spectra = spectra[:count]
        designs = designs[:count]
    return jnp.asarray(spectra, dtype=jnp.float32), jnp.asarray(designs, dtype=jnp.float32)


def _normalize_spectra(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D spectrum matrix, got shape {data.shape}")
    if data.shape[1] == 81:
        return data.astype(np.float32)
    if data.shape[0] == 81:
        return data.T.astype(np.float32)
    raise ValueError(f"Expected one spectrum dimension to be 81, got {data.shape}")


def _normalize_designs(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 4 and data.shape[1:] == (1, 12, 12):
        designs = data
    elif data.ndim == 3 and data.shape[-2:] == (12, 12):
        designs = data[:, None, :, :]
    elif data.ndim == 3 and data.shape[:2] == (12, 12):
        designs = np.transpose(data, (2, 0, 1))[:, None, :, :]
    elif data.ndim == 2 and data.shape[1] == 144:
        designs = data.reshape((-1, 1, 12, 12))
    elif data.ndim == 2 and data.shape == (12, 12):
        designs = data.reshape((1, 1, 12, 12))
    else:
        raise ValueError(f"Expected designs shaped [N,1,12,12], [N,12,12], [N,144], or [12,12,N], got {data.shape}")
    designs = (designs > 0.5).astype(np.float32)
    designs[:, 0, 5:7, 0] = 1.0
    return designs


def sample_antithetic_spectra(key, spectra: jnp.ndarray, population_size: int) -> jnp.ndarray:
    if population_size % 2 != 0:
        raise ValueError("population_size must be even for antithetic EGGROLL perturbations")
    pair_count = population_size // 2
    indices = jax.random.randint(key, (pair_count,), 0, spectra.shape[0])
    paired = spectra[indices]
    return jnp.repeat(paired, 2, axis=0)
