from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax.numpy")

from pixelant_eggroll.config import ScorerConfig
from pixelant_eggroll.scorers import ExternalEMScorer, build_scorer


def _write_fake_solver(path: Path, fail: bool = False) -> None:
    if fail:
        path.write_text("import sys\nsys.exit(7)\n", encoding="utf-8")
        return
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "import numpy as np",
                "import scipy.io",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--input', required=True)",
                "parser.add_argument('--output', required=True)",
                "parser.add_argument('--counter', required=True)",
                "args = parser.parse_args()",
                "counter = Path(args.counter)",
                "count = int(counter.read_text()) if counter.exists() else 0",
                "counter.write_text(str(count + 1))",
                "mat = scipy.io.loadmat(args.input)",
                "designs = np.asarray(mat['designs'])",
                "n = designs.shape[0]",
                "fill = designs.reshape(n, -1).mean(axis=1).astype(np.float32)",
                "s11_db = np.repeat(fill[:, None], 81, axis=1)",
                "valid = np.ones((n, 1), dtype=bool)",
                "error_message = np.empty((n, 1), dtype=object)",
                "error_message[:] = ''",
                "scipy.io.savemat(args.output, {'s11_db': s11_db, 'valid': valid, 'error_message': error_message})",
            ]
        ),
        encoding="utf-8",
    )


def test_external_em_scorer_scores_and_caches(tmp_path):
    script = tmp_path / "fake_solver.py"
    counter = tmp_path / "counter.txt"
    _write_fake_solver(script)
    command = f'python "{script}" --input "{{input_mat}}" --output "{{output_mat}}" --counter "{counter}"'
    scorer = ExternalEMScorer(
        solver_mode="air",
        command=command,
        work_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        timeout_seconds=30,
    )
    designs = np.zeros((2, 1, 12, 12), dtype=np.float32)
    designs[1, 0, :, :] = 1.0

    first = np.asarray(scorer.score(designs))
    second = np.asarray(scorer.score(designs))

    assert first.shape == (2, 81)
    assert np.allclose(first, second)
    assert scorer.last_valid.tolist() == [True, True]
    assert counter.read_text() == "1"


def test_external_em_scorer_requires_configured_command(tmp_path):
    scorer = ExternalEMScorer(
        solver_mode="air",
        work_dir=tmp_path,
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(NotImplementedError, match="not implemented/configured"):
        scorer.score(np.zeros((1, 1, 12, 12), dtype=np.float32))


def test_external_em_scorer_failed_command_maps_to_bad_spectrum(tmp_path):
    script = tmp_path / "fail_solver.py"
    _write_fake_solver(script, fail=True)
    scorer = ExternalEMScorer(
        solver_mode="air",
        command=f'python "{script}" --input "{{input_mat}}" --output "{{output_mat}}"',
        work_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        timeout_seconds=30,
        bad_spectrum_value=123.0,
    )

    spectrum = np.asarray(scorer.score(np.zeros((1, 1, 12, 12), dtype=np.float32)))

    assert spectrum.shape == (1, 81)
    assert np.all(spectrum == 123.0)
    assert scorer.last_valid.tolist() == [False]
    assert "MATLAB EM scorer failed" in scorer.last_error_message[0]


def test_build_external_em_scorer_from_config(tmp_path):
    scorer = build_scorer(
        ScorerConfig(
            kind="external-em",
            solver_mode="substrate",
            work_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            timeout_seconds=12,
            bad_spectrum_value=123.0,
        )
    )
    assert isinstance(scorer, ExternalEMScorer)
    assert scorer.solver_mode == "substrate"
    assert scorer.timeout_seconds == 12
