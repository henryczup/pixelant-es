"""Plot target, pretrained NN, and direct-ES S11 overlays for one target."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pixelant_eggroll.antenna_ops import hard_threshold_design
from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint, inverse_from_torch_checkpoint
from pixelant_eggroll.data import load_spectra_mat
from pixelant_eggroll.models_jax import bn_eval, forward_surrogate, leaky_relu, linear_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-mat", default="antenna_dataset.mat")
    parser.add_argument("--forward-checkpoint", default="Forward_model_for_tandem.pth")
    parser.add_argument("--inverse-checkpoint", default="inverse_tandem_model.pth")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def pure_inverse_forward(params, frozen_params, spectra: jnp.ndarray) -> jnp.ndarray:
    x = spectra
    for idx in range(len(frozen_params["hidden_dims"])):
        x = linear_eval(x, params[f"fc{idx}"])
        bn = {**frozen_params[f"bn{idx}"], **params[f"bn{idx}"]}
        x = leaky_relu(bn_eval(x, bn))
    return linear_eval(x, params["out"])


def load_best_es_design(run_dir: Path) -> jnp.ndarray:
    best_step = None
    best_mse = float("inf")
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mse = float(row["best_population_mse"])
            if mse < best_mse:
                best_mse = mse
                best_step = int(row["step"])
    if best_step is None:
        raise ValueError(f"No metrics found in {run_dir / 'metrics.csv'}")

    checkpoint = run_dir / "final_logits.npz"
    if checkpoint.exists():
        logits = jnp.asarray(np.load(checkpoint)["logits"])
        return hard_threshold_design(logits[None, :])

    raise FileNotFoundError(
        f"{checkpoint} was not found. Rerun compare_direct_design_es_vs_nn.py after the final-logits export patch."
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "s11_overlay.png"

    spectra = load_spectra_mat(args.spectra_mat)
    target = spectra[args.target_index : args.target_index + 1]
    surrogate_params = forward_surrogate_from_torch_checkpoint(args.forward_checkpoint)
    inv_frozen, inv_params, _scan, _es = inverse_from_torch_checkpoint(args.inverse_checkpoint)

    nn_logits = pure_inverse_forward(inv_params, inv_frozen, target)
    nn_design = hard_threshold_design(nn_logits)
    nn_pred = forward_surrogate(surrogate_params, nn_design)
    nn_mse = float(jax.device_get(jnp.mean((nn_pred - target) ** 2)))

    es_design = load_best_es_design(run_dir)
    es_pred = forward_surrogate(surrogate_params, es_design)
    es_mse = float(jax.device_get(jnp.mean((es_pred - target) ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(jax.device_get(target[0]), label="target", linewidth=2)
    axes[0].plot(jax.device_get(nn_pred[0]), label=f"NN generator MSE={nn_mse:.3f}")
    axes[0].plot(jax.device_get(es_pred[0]), label=f"Direct ES MSE={es_mse:.3f}")
    axes[0].set_title("S11 overlay")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("S11")
    axes[0].legend(fontsize=8)

    axes[1].imshow(jax.device_get(nn_design[0, 0]), cmap="gray_r", interpolation="nearest")
    axes[1].set_title("NN mask")
    axes[1].axis("off")

    axes[2].imshow(jax.device_get(es_design[0, 0]), cmap="gray_r", interpolation="nearest")
    axes[2].set_title("Direct ES mask")
    axes[2].axis("off")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
