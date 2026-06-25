"""Export PNGF-DBS optimizer logs into a target-conditioned generator dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pixelant_eggroll.pngf import export_pngf_dbs_records, parse_pngf_dbs_log, pngf_masks_are_projected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True, help="PNGF log files or glob patterns.")
    parser.add_argument("--output-npz", required=True, help="Output `.npz` path.")
    return parser.parse_args()


def _expand_patterns(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(char in pattern for char in "*?["):
            files.extend(sorted(path.parent.glob(path.name)))
        else:
            files.append(path)
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main() -> None:
    args = parse_args()
    logs = _expand_patterns(args.logs)
    if not logs:
        raise SystemExit("No PNGF log files matched --logs")
    records = []
    for log_path in logs:
        records.extend(parse_pngf_dbs_log(log_path, run_id=log_path.stem))
    export_pngf_dbs_records(records, args.output_npz)
    with np.load(args.output_npz, allow_pickle=False) as data:
        masks = data["masks"]
        accepted = data["accepted"]
        projected = pngf_masks_are_projected(masks)
    print(
        f"wrote {len(records)} PNGF records to {args.output_npz} "
        f"({int(np.sum(accepted))} accepted, {int(len(accepted) - np.sum(accepted))} rejected/initial); "
        f"projected={int(np.sum(projected))}/{len(projected)}"
    )


if __name__ == "__main__":
    main()
