# ES-Trained Hard-Binary Inverse Design: Findings Report

Date: 2026-06-15

## Executive Summary

We evaluated whether evolution strategies (ES), specifically an EGGROLL/HyperscaleES-style low-rank perturbation method, can train a one-shot inverse neural network for hard-binary pixelated antenna design without using gradients through the discretization or solver.

The main conclusion is:

```text
ES is better aligned than supervised inverse training for hard-discrete black-box inverse design,
but direct per-target pixel optimization is still much stronger than the current one-shot generator.
```

In the strongest longer surrogate run, ES reduced held-out hard-mask spectrum MSE from the random-initialized baseline:

```text
Initial test scorer MSE: 4.8564
ES generator, 500 steps: 3.0658
Policy gradient, 500 steps: 3.5623
Supervised inverse, 500 steps: 4.5866
```

However, direct BPSO over pixels on representative targets was much better:

```text
Representative-cut mean MSE:
Supervised inverse: 3.9476
ES generator:       2.0251
Policy gradient:    2.4010
BPSO pixels:        0.3095
```

This means the ES-trained generator is promising as a fast proposal model or initializer, but it is not yet good enough to replace local black-box optimization.

## Research Question

The central question was:

```text
Can a one-shot inverse neural network be trained better with ES when the downstream scorer/solver is non-differentiable?
```

This matters because practical EM design often contains non-differentiable steps:

```text
hard binary metal/no-metal decisions
layout projection
meshing
MATLAB/CST/HFSS/ADS solver calls
solver failures
fabrication-rule checks
```

Backpropagation through the solver is usually unavailable. ES only needs scalar fitness values, so it can train a generator against a true black-box scorer.

## Methods Compared

### Supervised Inverse Training

The supervised inverse network learns:

```text
target spectrum -> known dataset design
```

Loss:

```text
L_supervised = MSE(sigmoid(G_theta(S_target)), D_dataset)
```

This does not require solver gradients. But it optimizes design imitation, not spectrum matching.

### ST/Tandem Backprop

The tandem approach uses a frozen forward surrogate:

```text
target S11 -> generator -> ST activation -> surrogate -> spectrum loss
```

Loss:

```text
L_tandem =
MSE(F_surrogate(ST(G_theta(S_target))), S_target)
+ physical penalties
```

This works only when the scorer path is differentiable or approximated as differentiable.

### ES / EGGROLL Inverse Training

The ES generator path uses hard thresholding:

```text
theta_i = theta + sigma * eps_i
logits_i = G_theta_i(S_target)
D_i = hard_threshold(logits_i)
S_pred_i = scorer(D_i)
```

Fitness:

```text
fitness_i =
- MSE(S_pred_i, S_target)
- lambda_conn * connectivity_penalty(D_i)
- lambda_area * area_penalty(D_i)
- lambda_frag * fragmentation_penalty(D_i)
- lambda_rule * fabrication_rule_penalty(D_i)
```

Then:

```text
theta <- theta + lr * sum_i normalized_fitness_i * eps_i
```

No gradient passes through:

```text
hard_threshold
surrogate/scorer
EM solver
```

### Policy Gradient

The policy-gradient generator samples binary masks:

```text
p = sigmoid(G_theta(S_target) / temperature)
D ~ Bernoulli(p)
```

Loss:

```text
L_policy =
- advantage * log_prob_theta(D | S_target)
- beta_entropy * entropy(p)
```

This also avoids solver gradients, but suffered severe mode collapse in our runs.

### Direct BPSO Over Pixels

BPSO does not train a generator. It directly optimizes the hard mask for each target:

```text
min_D MSE(scorer(D), S_target)
```

This is slower at inference because it requires many solver calls per target, but it provides an important performance baseline.

## Experimental Setup

Dataset and checkpoints:

```text
antenna_dataset.mat
Forward_model_for_tandem.pth
inverse_tandem_model.pth
```

Primary scripts:

```text
compare_supervised_vs_es.py
compare_bpso_pixels.py
train_eggroll_inverse.py
score_designs_em.m
```

The frozen forward CNN surrogate was used as a black-box scorer for most experiments:

```text
hard binary mask -> frozen surrogate -> predicted S11
```

ES did not use surrogate gradients. The surrogate was used only as a fast stand-in for a non-differentiable EM solver.

The current hard-binary design is:

```text
12 x 12 binary metal mask
144 generator logits
feed pixels forced to metal
```

## Key Result 1: Supervised Inverse Training Learns Design Imitation, Not Spectrum Matching

Large supervised-vs-ES run:

```text
train size:       100000
validation size: 512
test size:       512
generator:       81 -> 256 -> 256 -> 144
steps:           300
ES population:   1024
```

Final results:

| Method | Val Design MSE | Val Scorer MSE | Test Design MSE | Test Scorer MSE | Uniqueness | Fill |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 0.3152 | 3.9468 | 0.3153 | 4.1165 | 0.977 | 0.486 |
| Supervised inverse | **0.2188** | 4.6543 | **0.2184** | 4.9672 | 0.012 | 0.9998 |
| ES inverse | 0.5006 | **3.4724** | 0.5025 | **3.6041** | **0.891** | 0.474 |

Interpretation:

```text
Supervised training improved design reconstruction MSE,
but it made hard-mask spectrum matching worse.
```

The supervised model collapsed toward nearly all-metal masks:

```text
fill:       0.9998
uniqueness: 0.012
```

This is consistent with the non-unique inverse problem. Many masks can yield similar spectra, so copying one dataset mask is not equivalent to solving inverse design.

## Key Result 2: ES Improves With Longer Training, Policy Gradient Collapses

Longer 500-step comparison:

```text
train size:       10000
validation size: 128
test size:       128
generator:       81 -> 128 -> 128 -> 144
population:      512
steps:           500
```

Final held-out test MSE:

| Method | Test Scorer MSE | Uniqueness | Fill |
|---|---:|---:|---:|
| Initial | 4.8564 | 0.9922 | 0.5184 |
| Supervised inverse | 4.5866 | 0.0547 | 0.9991 |
| ES generator | **3.0658** | **0.8906** | 0.4780 |
| Policy gradient | 3.5623 | 0.0078 | 0.4236 |

ES improved steadily:

```text
ES at 200 steps: 3.5662
ES at 500 steps: 3.0658
```

Policy gradient improved early but then collapsed and stopped improving:

```text
policy uniqueness at 500 steps: 0.0078
```

Figure:

![500-step test spectrum MSE](../eggroll_runs/supervised_es_policy_500/test_spectrum_mse.png)

Representative S11 cuts:

![500-step representative S11 cuts](../eggroll_runs/supervised_es_policy_500/representative_s11_cuts.png)

The representative cuts show ES generally improves over supervised and policy, but still misses some difficult targets.

## Key Result 3: ES Preserves More Mask Diversity Than Policy Gradient

ES does not explicitly optimize uniqueness, but it preserved diverse outputs:

```text
ES uniqueness:     0.8906
Policy uniqueness: 0.0078
Supervised:        0.0547
```

Likely reason:

```text
ES perturbs the whole generator function and updates toward perturbations that improve target-conditioned hard-mask behavior.
```

Policy gradient directly pushes Bernoulli probabilities toward sampled masks that got high reward. Once the logits saturate, sampling becomes nearly deterministic and exploration collapses.

Figure:

![Mask uniqueness](../eggroll_runs/supervised_es_policy_500/mask_uniqueness.png)

## Key Result 4: Direct BPSO Over Pixels Strongly Beats One-Shot Generators

We compared the trained inverse generators against direct BPSO on the same representative target spectra.

Mean MSE over six representative cuts:

| Method | Mean MSE |
|---|---:|
| Supervised inverse | 3.9476 |
| ES generator | 2.0251 |
| Policy gradient | 2.4010 |
| BPSO pixels | **0.3095** |

Per-target comparison:

| Target | Supervised | ES | Policy | BPSO Pixels |
|---:|---:|---:|---:|---:|
| 0 | 2.3289 | 0.3795 | 0.1821 | **0.0091** |
| 1 | 4.2080 | 1.3623 | 0.4528 | **0.0090** |
| 2 | 5.0511 | 0.4296 | 0.7809 | **0.0190** |
| 3 | 3.6201 | 1.3990 | 0.5543 | **0.1091** |
| 4 | 5.7568 | 11.6229 | 10.6425 | **1.5366** |
| 5 | 2.7208 | 1.5025 | 1.7936 | **0.1742** |

Figure:

![S11 cuts with BPSO](../eggroll_runs/bpso_pixels_vs_inverse_cuts/s11_cuts_with_bpso.png)

Generated masks:

![Hard masks with BPSO](../eggroll_runs/bpso_pixels_vs_inverse_cuts/hard_masks_with_bpso.png)

Interpretation:

```text
For 12x12 designs, per-target direct optimization is much stronger than the current one-shot generator.
```

This does not make the generator useless. It suggests the generator should be used as:

```text
a fast proposal model
or initializer for BPSO/CMA/DBS/local ES refinement
```

rather than the final optimizer.

## Key Result 5: Backprop Still Wins When A Good Differentiable Path Exists

In synthetic and surrogate-gradient experiments where a usable differentiable path existed, ST/backprop generally improved faster than ES from random initialization.

Observed pattern:

```text
If gradients are valid and not killed by hard thresholding, backprop is more sample efficient.
If the final design must be hard-discrete and scorer gradients are unavailable, ES is better aligned.
```

In scaling tests with very hard ST activations, ES became comparatively more useful as gradients vanished.

Important distinction:

```text
ES is not universally better than backprop.
ES is useful when the objective includes non-differentiable hard masks or black-box solvers.
```

## MATLAB EM Integration Status

We implemented a headless MATLAB scorer:

```text
score_designs_em.m
```

Modes:

```text
air:       fast approximate mode, 10-20 GHz
substrate: FR-4 substrate mode, 1-5 GHz
```

The Python scorer:

```text
ExternalEMScorer
```

does:

```text
write masks to .mat
call matlab -batch
read S11 output
cache results by mask/config hash
map solver failures to bad fitness
```

Smoke tests completed:

```text
air mode:       produced valid [1, 3] S11 output
substrate mode: produced valid [1, 3] S11 output
Python cache:   second call hit cache
```

This means the codebase is ready for small real-EM-budget experiments.

## Interpretation

The results support three claims.

### 1. ES Optimizes The Correct Objective Better Than Supervised Inverse Training

Supervised inverse training asks:

```text
Did the model copy the paired dataset mask?
```

ES asks:

```text
Does the generated hard mask score well for the requested spectrum?
```

For non-unique inverse EM design, the second question is more relevant.

### 2. One-Shot Inverse Generation Is Not Yet Sufficient

The current one-shot generator does improve with ES, but it does not produce high-quality designs compared with per-target BPSO.

The likely reason is that one-shot inverse design must amortize a difficult many-to-one inverse problem:

```text
many target spectra -> many good possible masks
```

This is harder than optimizing one mask for one target.

### 3. The Best Use Of The Generator Is As An Initializer

The most promising workflow is:

```text
target spectrum
-> ES-trained generator
-> initial hard mask
-> local BPSO/CMA/DBS/ES refinement
-> EM verification
```

This uses the neural network to reduce search time while preserving the robustness of direct black-box optimization.

## Recommended Next Experiments

### Experiment 1: Generator-Seeded BPSO

Compare:

```text
BPSO from random masks
BPSO initialized near supervised generator masks
BPSO initialized near ES generator masks
BPSO initialized near policy generator masks
```

Primary metric:

```text
best scorer/EM MSE after equal solver-call budget
```

This directly tests whether the generator provides useful starting points.

### Experiment 2: Small MATLAB Air-Solver ES Fine-Tune

Use:

```text
targets: 4-8
population: 8-16
steps: 5-20
solver: MATLAB air
```

Compare:

```text
random generator
warm-start ES generator
generator + local pixel refinement
```

Primary metric:

```text
real MATLAB air-solver S11 MSE
```

### Experiment 3: Air-To-Substrate Transfer

Use air solver as cheap exploratory scorer:

```text
air optimize -> substrate verify
```

Question:

```text
Does air-mode improvement correlate with substrate-mode improvement?
```

If yes, air mode can be a useful prefilter.

### Experiment 4: Latent/Manifold Search

Train a design autoencoder or VAE:

```text
mask -> latent z -> reconstructed mask
z -> predicted S11
```

Then optimize:

```text
z -> decoder -> hard mask -> scorer/EM
```

Compare:

```text
latent ES/CMA
pixel BPSO
one-shot generator
generator + local refinement
```

This may provide smoother search than raw pixels.

### Experiment 5: PNGF Or Other Fast Full-Wave Scorer

The recent precomputed numerical Green function approach could provide a fast, high-fidelity scorer for fixed design regions.

If available, use it as:

```text
generator -> hard mask -> PNGF scorer -> ES/BPSO fitness
```

This could reduce dependence on approximate neural surrogates while avoiding slow commercial solver calls.

## Limitations

1. Most comparisons used the frozen surrogate as the scorer, not full MATLAB/CST/HFSS EM.
2. ES did not use surrogate gradients, so the optimization structure matches black-box EM, but the scorer may still be inaccurate.
3. One-shot generator quality is currently modest.
4. Policy-gradient results may improve with better entropy schedules, baselines, or multi-sample objectives.
5. Results are for 12x12 single-layer antenna masks; multilayer/multiport passives will require more general layout representations.

## Bottom Line

The strongest conclusion is:

```text
ES/EGGROLL is a valid way to train hard-binary inverse generators against non-differentiable scorers,
and it is better aligned with inverse EM objectives than supervised mask reconstruction.
```

But:

```text
direct pixel optimization currently produces much better designs than one-shot generation.
```

Therefore, the next research direction should be:

```text
use ES-trained generators as proposal models,
then refine with direct black-box pixel optimization,
and evaluate with real EM.
```

This direction is directly relevant to practical EM/circuit workflows where the final scorer is MATLAB, CST, HFSS, ADS, PNGF, or measurement rather than a differentiable neural surrogate.

