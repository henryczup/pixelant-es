# Supervised Inverse NN vs ES-Trained Inverse NN

## Question

Can a one-shot inverse neural network be trained better with ES when the downstream solver is non-differentiable?

The comparison here is between:

1. **Supervised inverse backprop**
   - Uses paired dataset labels: `spectrum -> known binary design`.
   - Optimizes design reconstruction:
     `MSE(sigmoid(G(S)), D_dataset)`.
   - Does not need solver gradients, but does not optimize the solver objective directly.

2. **ES / EGGROLL inverse training**
   - Uses the same inverse generator architecture and the same target spectra.
   - Generates hard binary masks:
     `D = hard_threshold(G(S))`.
   - Optimizes scalar scorer fitness:
     `-MSE(scorer(D), S)`.
   - Does not need gradients through the threshold or scorer.

The important distinction is that supervised inverse training tries to copy a dataset design, while ES training tries to produce any hard design that scores well for the requested spectrum.

## Setup

Assets:

- Dataset: `antenna_dataset.mat`
- Scorer: `Forward_model_for_tandem.pth`
- Script: `compare_supervised_vs_es.py`

The scorer in this run is the frozen forward surrogate used as a black-box evaluator. ES does not use gradients from it. This gives a fast stand-in for a non-differentiable EM solver and lets us run larger comparisons before spending MATLAB EM budget.

Both methods start from the exact same random inverse generator weights:

```text
Initial: 81 -> 256 -> 256 -> 144
```

The generator maps target S11 spectra to 144 logits, then hard-thresholds into a 12x12 binary mask for scorer evaluation.

## Command

```bash
python compare_supervised_vs_es.py \
  --dataset antenna_dataset.mat \
  --forward-checkpoint Forward_model_for_tandem.pth \
  --scorer surrogate \
  --train-size 100000 \
  --val-size 512 \
  --test-size 512 \
  --hidden-dims 256,256 \
  --supervised-steps 300 \
  --es-steps 300 \
  --eval-every 50 \
  --batch-size 512 \
  --population-size 1024 \
  --supervised-lr 0.001 \
  --es-lr 0.01 \
  --es-sigma 0.2 \
  --output-dir eggroll_runs/supervised_vs_es_large_test_300
```

## Metrics

### Solver/Scorer Spectrum MSE

This is the objective that matters for inverse design:

```text
S_target -> G(S_target) -> hard mask D -> scorer(D) = S_pred
MSE(S_pred, S_target)
```

Lower means the generated hard antenna mask better matches the requested S11 spectrum.

### Design Reconstruction MSE

This compares the generated design to the dataset's original design:

```text
MSE(predicted_design, D_dataset)
```

Lower means the model copied the dataset mask better. This is not necessarily equivalent to better inverse design, because the inverse map is one-to-many.

### Mask Uniqueness

Fraction of unique hard masks in the validation batch. Low uniqueness indicates mode collapse.

### Metal Fill

Mean fraction of metal pixels in generated hard masks.

## Results

| Method | Val Design MSE | Val Scorer MSE | Test Design MSE | Test Scorer MSE | Uniqueness | Fill |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 0.3152 | 3.9468 | 0.3153 | 4.1165 | 0.977 | 0.486 |
| Supervised inverse | **0.2188** | 4.6543 | **0.2184** | 4.9672 | 0.012 | 0.9998 |
| ES inverse | 0.5006 | **3.4724** | 0.5025 | **3.6041** | **0.891** | 0.474 |

Full CSV:

```text
eggroll_runs/supervised_vs_es_large_test_300/supervised_vs_es.csv
```

## Plots

Validation scorer MSE:

![Validation Scorer MSE](../eggroll_runs/supervised_vs_es_large_test_300/supervised_vs_es.png)

Held-out test scorer MSE:

![Test Scorer MSE](../eggroll_runs/supervised_vs_es_large_test_300/test_spectrum_mse.png)

Design reconstruction MSE:

![Design MSE](../eggroll_runs/supervised_vs_es_large_test_300/design_mse.png)

Held-out test design reconstruction MSE:

![Test Design MSE](../eggroll_runs/supervised_vs_es_large_test_300/test_design_mse.png)

Mask uniqueness:

![Mask Uniqueness](../eggroll_runs/supervised_vs_es_large_test_300/mask_uniqueness.png)

Metal fill:

![Metal Fill](../eggroll_runs/supervised_vs_es_large_test_300/metal_fill.png)

## Interpretation

The supervised inverse model successfully reduces design reconstruction loss:

```text
test design MSE: 0.3153 -> 0.2184
```

But this does not translate into better hard-mask spectrum matching:

```text
test scorer MSE: 4.1165 -> 4.9672
```

The supervised model also collapses:

```text
mask uniqueness: 0.977 -> 0.012
metal fill:      0.486 -> 0.9998
```

This is consistent with the one-to-many inverse design problem. Many different antenna masks can produce similar spectra, but supervised design MSE forces the network toward the specific dataset mask. With ambiguous targets, MSE can reward average or degenerate masks that do not score well after hard thresholding.

The ES-trained inverse model moves away from the dataset designs:

```text
test design MSE: 0.3153 -> 0.5025
```

But it improves the actual inverse-design objective:

```text
test scorer MSE: 4.1165 -> 3.6041
```

It also keeps useful diversity:

```text
mask uniqueness: 0.977 -> 0.891
metal fill:      0.486 -> 0.474
```

This suggests ES is better aligned with the real black-box objective:

```text
produce any hard mask whose solved/scored spectrum matches the target
```

rather than:

```text
copy the particular dataset mask paired with that spectrum
```

## Conclusion

For one-shot inverse NN training without gradients from the solver, ES/EGGROLL is the better method in this experiment when judged by solver/scorer spectrum MSE.

Supervised inverse backprop is better at design imitation, but design imitation is the wrong objective for non-unique inverse antenna synthesis. It can reduce pixel-level design error while degrading hard-mask solver performance.

The practical workflow should be:

```text
1. Optionally use supervised data or surrogate/ST training for cheap initialization.
2. Use ES/EGGROLL with hard masks to optimize the actual scorer or EM solver objective.
3. Evaluate final designs by held-out scorer or MATLAB EM, not design reconstruction MSE.
```

## Limitations and Next Steps

- This run used the frozen surrogate as a black-box scorer, not MATLAB EM. ES did not use scorer gradients, so the optimization structure matches EM usage, but the scorer is still an approximation.
- The next experiment should repeat the comparison with `--scorer external-em --solver-mode air` on a small EM budget.
- After the air-mode EM path is stable, repeat with `--solver-mode substrate` for the FR-4 solver.
- A hybrid method should also be tested:

```text
supervised pretrain -> ES fine-tune with hard masks and EM/scorer fitness
```

This may combine the dataset prior from supervised learning with the objective correctness of ES.
