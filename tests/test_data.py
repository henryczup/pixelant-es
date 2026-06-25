import numpy as np
import pytest
import scipy.io

jnp = pytest.importorskip("jax.numpy")

from pixelant_eggroll.data import load_paired_antenna_mat, load_spectra_mat


def test_load_spectra_mat_transposes_existing_dataset_layout(tmp_path):
    path = tmp_path / "spectra.mat"
    scipy.io.savemat(path, {"YTrain": np.ones((81, 3), dtype=np.float32)})

    spectra = load_spectra_mat(path)

    assert spectra.shape == (3, 81)


def test_load_spectra_mat_accepts_row_major_layout(tmp_path):
    path = tmp_path / "spectra.mat"
    scipy.io.savemat(path, {"specd": np.ones((4, 81), dtype=np.float32)})

    spectra = load_spectra_mat(path)

    assert spectra.shape == (4, 81)


def test_load_paired_antenna_mat_normalizes_existing_dataset_layout(tmp_path):
    path = tmp_path / "paired.mat"
    x = np.zeros((3, 1, 12, 12), dtype=np.uint8)
    y = np.ones((81, 3), dtype=np.float32)
    scipy.io.savemat(path, {"XTrain1": x, "YTrain": y})

    spectra, designs = load_paired_antenna_mat(path)

    assert spectra.shape == (3, 81)
    assert designs.shape == (3, 1, 12, 12)
    assert designs[:, 0, 5:7, 0].sum() == 6


def test_load_paired_antenna_mat_accepts_flat_designs(tmp_path):
    path = tmp_path / "paired.mat"
    x = np.zeros((2, 144), dtype=np.uint8)
    y = np.ones((2, 81), dtype=np.float32)
    scipy.io.savemat(path, {"designs": x, "spec": y})

    spectra, designs = load_paired_antenna_mat(path)

    assert spectra.shape == (2, 81)
    assert designs.shape == (2, 1, 12, 12)
