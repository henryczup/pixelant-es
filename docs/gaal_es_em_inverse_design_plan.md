# GAAL-Style Active EM Learning With ES-Trained Inverse Generators

## Summary

This plan adapts Generative Adversarial Active Learning (GAAL) to EM passive and antenna inverse design.

The core idea is:

```text
generator proposes informative hard layouts
non-differentiable EM solver labels them with S-parameters
surrogate learns from the growing EM dataset
inverse generator learns to propose better layouts
active loop repeats
```

In the original GAAL framing, a generator synthesizes informative inputs and a human oracle labels them. In this EM workflow:

```text
query/input       = hard binary or multilayer layout
oracle            = MATLAB / CST / HFSS / measured EM result
learner           = surrogate ensemble plus inverse generator
label             = simulated multiport S-parameters
active criterion  = promising + uncertain + diverse + valid
```

The surrogate should be trained with backprop. The inverse generator can be trained with backprop when using differentiable surrogate paths, and with ES when the scorer is hard-thresholded or a non-differentiable EM solver.

## Goals

1. Build an active EM data-generation loop that spends solver calls on useful designs rather than random masks.
2. Train a surrogate ensemble that is accurate in the regions where optimizers and generators search.
3. Train an inverse generator that maps target S-parameters/specs to hard layouts.
4. Support non-differentiable EM solver feedback through ES.
5. Compare one-shot generators against direct per-target optimizers such as BPSO/CMA and generator-seeded local search.
6. Extend from the current 12x12 one-layer antenna masks to multilayer and multiport passive layouts.

## Non-Goals

1. Do not rely on GAN training as the only generator method. Conditional VAE, diffusion-style models, CNN decoders, and ES-trained generators are all acceptable.
2. Do not assume one-shot inverse generation will beat direct optimization. Based on current experiments, the generator is more likely to be useful as a proposal model or optimizer initializer.
3. Do not train the surrogate itself with ES unless gradients are unavailable. Surrogate training has clean supervised gradients and should use backprop.

## System Components

### 1. EM Dataset

Each sample should contain:

```text
layout_id
layout tensor
stackup/material config
port config
frequency grid
S-parameter tensor
solver metadata
validity flag
error message
```

For the current antenna experiments:

```text
layout: [1, 12, 12] binary metal mask
ports: fixed feed pixels
S: [81] S11 dB
solver modes: air, substrate
```

For generalized passives:

```text
metal: [L, H, W]
via:   [L-1, H, W]
ports: fixed or templated port pads
S:     [F, P, P, 2] for complex real/imag S-parameters
```

### 2. Surrogate Ensemble

Use an ensemble instead of one surrogate:

```text
F_1(D) -> S_pred
F_2(D) -> S_pred
...
F_K(D) -> S_pred
```

The ensemble gives:

```text
mean prediction:      mean_k F_k(D)
uncertainty estimate: var_k F_k(D)
```

Training loss:

```text
loss_surrogate =
MSE(S_pred, S_EM)
+ optional passivity/reciprocity/smoothness penalties
```

For multiport passive devices, prefer complex S-parameters:

```text
S_target[f, i, j, :] = [Re(Sij), Im(Sij)]
```

dB-only objectives can still be used for reporting/spec violations.

### 3. Inverse Generator

The inverse generator maps a target response or device spec to layout logits:

```text
G_theta(target_spec, ports, stackup, latent) -> layout logits
```

Current antenna form:

```text
G_theta(S11_target) -> logits [144]
D = hard_threshold(logits) -> [1, 12, 12]
```

Generalized form:

```text
G_theta(S_target, W_target, metadata, latent)
    -> metal_logits [L, H, W]
    -> via_logits   [L-1, H, W]
```

Then apply a non-differentiable hard projection:

```text
D_raw = threshold(logits)
D = project_to_valid_layout(D_raw, ports, stackup, design_rules)
```

The projection should:

```text
force port/feed pixels
force ground/reference structures if required
enforce keepout regions
remove illegal vias
snap to grid
apply min trace / min gap / via constraints
```

### 4. Query Generator

The query generator produces candidate layouts to send to the EM solver.

It can be one or more of:

```text
inverse generator outputs
latent-space perturbations around inverse generator outputs
BPSO/CMA local search around generator outputs
random/diverse exploration
uncertainty-seeking generator
```

This does not have to be the same model as the inverse generator. In early versions, simple candidate pools are easier and more robust than adversarial GAN training.

### 5. EM Scorer

Use the existing MATLAB scorer interface first:

```text
input_mat:
  designs
  freq
  solver_mode
  options

output_mat:
  s11_db or S_complex
  valid
  error_message
  elapsed_seconds
```

Current modes:

```text
air:       fast approximate scorer
substrate: slower FR-4 scorer
```

Later modes:

```text
CST
HFSS
measurement import
multiport PCB passive solver
```

Failures should not crash training. They should produce:

```text
valid = false
bad fitness / bad spectrum value
cached failure record
```

## Active Learning Loop

One active cycle:

```text
1. Train/retrain surrogate ensemble on all EM-labeled data.

2. Train or fine-tune inverse generator cheaply against the surrogate:
   target spec -> generator -> hard layout -> surrogate ensemble -> fitness

3. Generate a large candidate pool:
   - generator proposals
   - generator + noise
   - BPSO/CMA refinements
   - uncertainty-seeking candidates
   - random/diverse candidates

4. Score candidates with acquisition function:
   promising + uncertain + diverse + valid

5. Select top K candidates under diversity constraints.

6. Run real EM solver on selected candidates.

7. Add EM results to dataset.

8. Optionally do small ES fine-tuning of inverse generator directly against EM.

9. Evaluate on held-out target specs using real EM where budget allows.
```

## Acquisition Function

Candidate layouts should not be selected only because the surrogate predicts good performance. That risks exploiting surrogate errors.

Use:

```text
A(D, S_target) =
  w_perf   * target_relevance(D, S_target)
+ w_uncert * surrogate_uncertainty(D)
+ w_div    * diversity_score(D, dataset_or_batch)
- w_rule   * rule_penalty(D)
- w_fail   * predicted_failure_penalty(D)
```

Where:

```text
target_relevance(D, S_target) =
- weighted_response_error(mean_F(D), S_target)

surrogate_uncertainty(D) =
mean variance across surrogate ensemble predictions

diversity_score(D) =
distance from already selected candidates and existing EM dataset
```

For multiple target specs in one cycle:

```text
A(D, spec) =
  target-specific acquisition
```

Then select a balanced batch across specs.

## ES Training Of The Inverse Generator

ES is used when the training path includes non-differentiable operations:

```text
hard threshold
layout projection
MATLAB/CST/HFSS solver
meshing failures
discrete topology changes
```

For each ES step:

```text
theta_i = theta + sigma * eps_i

logits_i = G_theta_i(S_target)
D_i = project_to_valid_layout(hard_threshold(logits_i))

S_i = scorer(D_i)  # surrogate or real EM

fitness_i =
- weighted_response_error(S_i, S_target)
- lambda_conn * connectivity_penalty(D_i)
- lambda_area * area_penalty(D_i)
- lambda_frag * fragmentation_penalty(D_i)
- lambda_rule * fabrication_rule_penalty(D_i)
```

Then update:

```text
theta <- theta + lr * sum_i normalized_fitness_i * eps_i
```

Use antithetic perturbations:

```text
theta_plus  = theta + sigma * eps
theta_minus = theta - sigma * eps
```

The pair contributes:

```text
(fitness_plus - fitness_minus) * eps
```

Recommended training phases:

```text
Phase A: ES-surrogate training, large population, many steps.
Phase B: ES-EM fine-tuning, small population, few steps.
Phase C: add EM results to dataset and retrain surrogate.
```

## Fitness For Filters, Couplers, And Passives

Use a general weighted multiport response objective:

```text
response_error =
mean_f,i,j W[f,i,j] * |S_pred[f,i,j] - S_target[f,i,j]|^2
```

For spec-based design, use hinge penalties.

### 2-Port Filter

```text
fitness =
- passband_insertion_loss_error(S21)
- return_loss_error(S11, S22)
- stopband_rejection_error(S21)
- reciprocity_error(S12, S21)
- fabrication_penalties
```

### Quadrature Coupler

```text
fitness =
- coupling_error(|S21|, -3 dB)
- coupling_error(|S31|, -3 dB)
- isolation_error(|S41|, threshold)
- match_error(|S11|, threshold)
- phase_error(angle(S31) - angle(S21), 90 deg)
- fabrication_penalties
```

### Power Divider

```text
fitness =
- split_balance_error(|S21|, |S31|)
- insertion_loss_error(|S21|, |S31|)
- isolation_error(|S23|)
- match_error(S11, S22, S33)
- fabrication_penalties
```

### Matching Network

```text
fitness =
- return_loss_error(S11)
- optional insertion_loss_error(S21)
- bandwidth_error
- fabrication_penalties
```

## Latent-Space Variant

The strongest version likely uses a learned design manifold:

```text
encoder: D -> z
decoder: z -> D_hat
latent surrogate: z -> S_pred
```

Train with backprop:

```text
loss =
BCE(D_hat, D)
+ alpha * MSE(S_pred, S_EM)
+ beta * KL or latent prior penalty
```

Then optimize per target:

```text
z_i = z + sigma * eps_i
D_i = hard_threshold(decoder(z_i))
S_i = scorer(D_i)
fitness_i = -response_error(S_i, S_target) - penalties
z <- ES/CMA update
```

Final evaluation must always be:

```text
decoded hard layout -> forward surrogate or real EM
```

Do not trust only:

```text
z -> latent surrogate
```

because the latent surrogate may predict a response that the decoded hard layout does not realize.

## Comparisons To Run

For each target set, compare:

```text
1. Supervised inverse generator.
2. ST/tandem generator through surrogate.
3. ES generator through surrogate.
4. Policy-gradient generator through surrogate.
5. Direct BPSO over pixels/layout variables.
6. Direct CMA/ES over pixels or logits.
7. Latent-space ES/CMA.
8. Generator output + BPSO refinement.
9. Generator output + latent refinement.
10. ES-EM fine-tuned generator.
```

Primary metric:

```text
real EM-evaluated response error after fixed EM-call budget
```

Secondary metrics:

```text
surrogate response error
mask/layout uniqueness
metal fill ratio
validity rate
solver failure rate
best/median/worst target error
time per successful design
number of EM calls
```

## Milestones

### Milestone 1: Current 12x12 Antenna Active Loop

Scope:

```text
one layer
one port
S11 only
MATLAB air solver
small target batch
```

Tasks:

```text
1. Build persistent EM dataset format.
2. Train surrogate ensemble from EM dataset.
3. Generate candidates from inverse generator and BPSO.
4. Rank candidates by performance + uncertainty + diversity.
5. Run MATLAB air solver on selected candidates.
6. Retrain surrogate.
7. Report real air-solver improvement over cycles.
```

Success:

```text
active-selected EM data improves held-out EM prediction faster than random EM data
```

### Milestone 2: ES-EM Generator Fine-Tuning

Scope:

```text
same 12x12 antenna
MATLAB air solver
small ES population
few targets
```

Tasks:

```text
1. Warm-start generator from supervised/ST/ES-surrogate training.
2. Run ES directly against MATLAB air solver.
3. Cache all EM calls.
4. Add EM results to dataset.
5. Compare pre/post fine-tune real EM error.
```

Success:

```text
ES-EM fine-tuning improves real EM error for at least some targets under a fixed EM-call budget
```

### Milestone 3: Substrate Solver Validation

Scope:

```text
FR-4 substrate solver
smaller EM budget
top designs from air/surrogate loop
```

Tasks:

```text
1. Score air-optimized candidates in substrate mode.
2. Measure surrogate/air/substrate mismatch.
3. Add substrate results to dataset.
4. Train substrate-specific or multi-fidelity surrogate.
```

Success:

```text
identify whether air solver is a useful prefilter for substrate designs
```

### Milestone 4: Generalized Multiport LayoutSpec

Scope:

```text
multilayer metal/via tensors
multiport S-parameter tensors
design-rule projection
```

Tasks:

```text
1. Extend LayoutSpec for metal layers, via layers, ports, stackup, rules.
2. Extend hard threshold and projection functions.
3. Extend scorer output from [N, 81] to [N, F, P, P, 2].
4. Implement weighted multiport response loss.
```

Success:

```text
same active-learning and ES machinery works for non-antenna passive specs
```

### Milestone 5: Latent Manifold Search

Scope:

```text
autoencoder or VAE design manifold
latent ES/CMA
decoded hard-layout verification
```

Tasks:

```text
1. Train design autoencoder plus latent response head.
2. Optimize z for target S-parameters.
3. Decode and verify hard layouts through surrogate/EM.
4. Compare latent search against direct BPSO and generator-only outputs.
```

Success:

```text
latent search beats direct pixel search under equal solver-call or scorer-call budget
```

## Implementation Interfaces

### EM Dataset Record

```python
{
    "layout": np.ndarray,
    "s_params": np.ndarray,
    "freq_hz": np.ndarray,
    "stackup": dict,
    "ports": list[dict],
    "solver": dict,
    "valid": bool,
    "error_message": str,
}
```

### Candidate Record

```python
{
    "layout": np.ndarray,
    "source": "generator|bpso|random|latent|uncertainty",
    "target_id": str,
    "predicted_s": np.ndarray,
    "predicted_error": float,
    "uncertainty": float,
    "diversity": float,
    "rule_penalty": float,
    "acquisition": float,
}
```

### Active Cycle CLI

Proposed command:

```bash
python active_em_gaal_cycle.py \
  --dataset eggroll_runs/em_dataset \
  --targets targets.mat \
  --solver-mode air \
  --cycle-count 5 \
  --candidate-count 2048 \
  --em-query-count 32 \
  --surrogate-ensemble-size 5 \
  --generator-checkpoint inverse_generator.npz \
  --output-dir eggroll_runs/active_gaal_air
```

### ES-EM Fine-Tune CLI

Proposed command:

```bash
python train_eggroll_inverse.py \
  --scorer external-em \
  --solver-mode air \
  --population-size 8 \
  --steps 10 \
  --rank 1 \
  --sigma 0.1 \
  --warm-start inverse_generator.npz \
  --cache-dir eggroll_runs/em_cache_air
```

## Risks And Mitigations

### Surrogate Exploitation

Risk:

```text
generator or BPSO finds masks that fool the surrogate
```

Mitigation:

```text
use surrogate ensemble uncertainty
send high-uncertainty promising designs to EM
evaluate winners with real EM
retrain surrogate with exploited examples
```

### Mode Collapse

Risk:

```text
generator produces same layout for many targets
```

Mitigation:

```text
diversity acquisition term
latent noise input
multi-sample generator outputs per target
entropy or uniqueness regularization
generator-seeded local search
```

### EM Budget Explosion

Risk:

```text
ES-EM needs too many solver calls
```

Mitigation:

```text
use surrogate for most training
small EM populations
cache every solver call
rank candidates before EM
multi-fidelity air -> substrate -> CST/HFSS funnel
```

### Invalid Layouts

Risk:

```text
solver fails due to disconnected, unmeshable, or illegal structures
```

Mitigation:

```text
hard projection
design-rule penalties
failure classifier
bad-fitness handling
cache failures
```

## Recommended First Experiment

Use the existing 12x12 antenna problem:

```text
targets: 8 held-out S11 curves
solver: MATLAB air
initial data: downloaded dataset + small number of real air re-scores
surrogate: ensemble of 3 forward CNNs
candidate sources:
  - ES generator proposals
  - policy generator proposals
  - direct BPSO proposals
  - random masks
selection:
  top 16 by acquisition with diversity filtering
cycles: 3
```

Report:

```text
1. held-out EM prediction error after each cycle
2. best real EM design error per target
3. surrogate vs EM mismatch on selected candidates
4. acquisition source breakdown
5. representative S11 cuts and hard masks
6. EM calls spent
```

The immediate question:

```text
Does active EM selection improve the surrogate and final designs faster than random EM sampling?
```

The second question:

```text
Does the inverse generator help when used as a proposal source or BPSO initializer?
```

