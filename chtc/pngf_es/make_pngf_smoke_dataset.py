"""Create a tiny reachable PNGF target dataset by scoring random masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pixelant_eggroll.pngf import project_pngf_center_fed_mask_np
from pixelant_eggroll.scorers import PNGFScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer-command", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fill-prob", type=float, default=0.45)
    parser.add_argument("--cache-dir", default="results/pngf_cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    raw = (rng.random((args.count, 21, 21)) < args.fill_prob).astype(np.float32)
    masks = project_pngf_center_fed_mask_np(raw)
    scorer = PNGFScorer(
        command=args.scorer_command,
        work_dir=Path.cwd(),
        cache_dir=Path(args.cache_dir),
        timeout_seconds=3600.0,
        bad_spectrum_value=1.0e6,
    )
    targets = np.asarray(scorer.score(masks[:, None, :, :]), dtype=np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        masks=masks.astype(np.float32),
        targets=targets,
        valid=scorer.last_valid,
        error_message=np.asarray(scorer.last_error_message),
    )
    print(f"wrote {output} with {args.count} masks; valid={int(np.sum(scorer.last_valid))}/{len(scorer.last_valid)}")
    if not np.all(scorer.last_valid):
        raise SystemExit("PNGF smoke dataset scoring had invalid rows")


if __name__ == "__main__":
    main()
