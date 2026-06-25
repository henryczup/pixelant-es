from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax.numpy")

from pixelant_eggroll.config import ScorerConfig
from pixelant_eggroll.scorers import PNGFScorer, build_scorer


def _write_fake_pngf(path: Path, fail: bool = False) -> None:
    if fail:
        path.write_text("import sys\nsys.exit(9)\n", encoding="utf-8")
        return
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "import numpy as np",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--input', required=True)",
                "parser.add_argument('--output', required=True)",
                "parser.add_argument('--counter', required=True)",
                "args = parser.parse_args()",
                "counter = Path(args.counter)",
                "count = int(counter.read_text()) if counter.exists() else 0",
                "counter.write_text(str(count + 1))",
                "with np.load(args.input, allow_pickle=False) as data:",
                "    masks = data['masks'].astype(np.float32)",
                "n = masks.shape[0]",
                "fill = masks.reshape(n, -1).mean(axis=1).astype(np.float32)",
                "scale = np.arange(1, 6, dtype=np.float32)[None, :]",
                "s11_re = fill[:, None] / scale",
                "s11_im = -0.1 * fill[:, None] * scale",
                "directivity = 10.0 + fill[:, None] * scale",
                "valid = np.ones((n,), dtype=bool)",
                "error_message = np.array([''] * n)",
                "np.savez_compressed(args.output, s11_re=s11_re, s11_im=s11_im, directivity=directivity, valid=valid, error_message=error_message)",
            ]
        ),
        encoding="utf-8",
    )


def test_pngf_scorer_scores_and_caches(tmp_path):
    script = tmp_path / "fake_pngf.py"
    counter = tmp_path / "counter.txt"
    _write_fake_pngf(script)
    command = f'python "{script}" --input "{{input_npz}}" --output "{{output_npz}}" --counter "{counter}"'
    scorer = PNGFScorer(command=command, work_dir=tmp_path, cache_dir=tmp_path / "cache", timeout_seconds=30)
    masks = np.zeros((2, 1, 21, 21), dtype=np.float32)
    masks[1, 0, 0, 0] = 1.0

    first = np.asarray(scorer.score(masks))
    second = np.asarray(scorer.score(masks))

    assert first.shape == (2, 15)
    assert np.allclose(first, second)
    assert scorer.last_valid.tolist() == [True, True]
    assert scorer.last_s11_complex.shape == (2, 5)
    assert counter.read_text() == "1"


def test_pngf_scorer_requires_configured_command(tmp_path):
    scorer = PNGFScorer(work_dir=tmp_path, cache_dir=tmp_path / "cache")

    with pytest.raises(NotImplementedError, match="Provide command"):
        scorer.score(np.zeros((1, 1, 21, 21), dtype=np.float32))


def test_pngf_scorer_failed_command_maps_to_bad_target(tmp_path):
    script = tmp_path / "fail_pngf.py"
    _write_fake_pngf(script, fail=True)
    scorer = PNGFScorer(
        command=f'python "{script}" --input "{{input_npz}}" --output "{{output_npz}}"',
        work_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        timeout_seconds=30,
        bad_spectrum_value=321.0,
    )

    target = np.asarray(scorer.score(np.zeros((1, 1, 21, 21), dtype=np.float32)))

    assert target.shape == (1, 15)
    assert np.all(target == 321.0)
    assert scorer.last_valid.tolist() == [False]
    assert "PNGF scorer failed" in scorer.last_error_message[0]


def test_build_pngf_scorer_from_config(tmp_path):
    scorer = build_scorer(
        ScorerConfig(
            kind="pngf",
            work_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            timeout_seconds=12,
            bad_spectrum_value=123.0,
        )
    )
    assert isinstance(scorer, PNGFScorer)
    assert scorer.spectrum_dim == 15
    assert scorer.timeout_seconds == 12
