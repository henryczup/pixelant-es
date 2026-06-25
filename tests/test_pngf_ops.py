import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from pixelant_eggroll.pngf import (
    PNGF_TARGET_DIM,
    center_fed_symmetric_flip_indices,
    export_pngf_dbs_records,
    hard_project_pngf_center_fed,
    pack_pngf_targets,
    parse_pngf_dbs_log,
    pngf_masks_are_projected,
    pngf_paper_objective,
    pngf_target_errors,
    project_pngf_center_fed_mask_np,
)


def _perf(objective: float) -> str:
    values = [objective, objective + 1.0]
    for idx in range(5):
        values.extend([0.1 * idx, -0.01 * idx, 10.0 + idx])
    return " ".join(str(value) for value in values)


def test_hard_pngf_projection_enforces_symmetry_and_center_feed():
    logits = jnp.full((1, 1, 21, 21), -1.0)
    logits = logits.at[0, 0, 0, 1].set(5.0)
    mask = hard_project_pngf_center_fed(logits)

    assert mask.shape == (1, 1, 21, 21)
    assert mask[0, 0, 0, 1] == 1.0
    assert mask[0, 0, 0, 19] == 1.0
    assert mask[0, 0, 20, 1] == 1.0
    assert mask[0, 0, 20, 19] == 1.0
    assert mask[0, 0, 10, 9] == 1.0
    assert mask[0, 0, 10, 11] == 1.0
    assert mask[0, 0, 10, 10] == 0.0
    assert mask[0, 0, 9, 10] == 0.0
    assert mask[0, 0, 11, 10] == 0.0


def test_numpy_projection_and_flip_indices():
    mask = np.zeros((21, 21), dtype=np.float32)
    mask[3, 4] = 1.0
    projected = project_pngf_center_fed_mask_np(mask)

    assert projected.shape == (1, 21, 21)
    assert np.all(pngf_masks_are_projected(projected))
    assert projected[0, 3, 4] == 1.0
    assert projected[0, 3, 16] == 1.0
    assert projected[0, 17, 4] == 1.0
    assert projected[0, 17, 16] == 1.0
    assert center_fed_symmetric_flip_indices(4 + 3 * 21) == (67, 79, 361, 373)


def test_pngf_target_pack_and_errors():
    s11 = np.ones((2, 5), dtype=np.complex64) * (0.2 - 0.1j)
    directivity = np.ones((2, 5), dtype=np.float32) * 12.0
    targets = pack_pngf_targets(s11=s11, directivity=directivity)
    errors = pngf_target_errors(jnp.asarray(targets), jnp.asarray(targets), beta=2.0)

    assert targets.shape == (2, PNGF_TARGET_DIM)
    assert np.allclose(np.asarray(errors.total), 0.0)
    assert np.all(pngf_paper_objective(targets) > 0.0)


def test_parse_and_export_pngf_dbs_log(tmp_path):
    start = "." * (21 * 21)
    log = tmp_path / "log_7.txt"
    log.write_text(
        "\n".join(
            [
                "123 3",
                start,
                f"0 -1 -1 {_perf(100.0)}",
                f"1 0 1 {_perf(90.0)}",
                f"2 1 0 {_perf(95.0)}",
                start,
            ]
        ),
        encoding="utf-8",
    )

    records = parse_pngf_dbs_log(log)
    assert len(records) == 3
    assert records[1]["accepted"] is True
    assert records[2]["accepted"] is False
    assert records[1]["target"].shape == (PNGF_TARGET_DIM,)
    assert records[1]["mask"][0, 0] == 1.0
    assert records[1]["mask"][20, 20] == 1.0

    output = tmp_path / "dataset.npz"
    export_pngf_dbs_records(records, output)
    with np.load(output, allow_pickle=False) as data:
        assert data["masks"].shape == (3, 21, 21)
        assert data["targets"].shape == (3, PNGF_TARGET_DIM)
        assert data["accepted"].tolist() == [False, True, False]
