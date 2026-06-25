# Pixelated ES Passive Generator for FALCON-Style Circuit Design

## Purpose

This document describes how a pixelated passive-layout generator trained with evolution strategies (ES) could be combined with the FALCON framework for fully automated, layout-constrained analog/RF circuit design.

FALCON currently uses:

```text
target circuit specs
-> topology classifier
-> GNN forward model
-> gradient-based parameter inference
-> analytic layout-aware passive costs
```

The proposed extension uses:

```text
target circuit specs
-> topology classifier
-> active-device sizing / circuit scaffold
-> pixelated passive generator
-> EM/circuit black-box scorer
-> ES update
```

The main motivation is that FALCON's GNN-gradient path works well for continuous schematic parameters, but mm-wave passives, transitions, couplers, matching networks, and distributed layout effects are often geometry-dominated and hard to represent with simple differentiable formulas.

## FALCON Baseline

FALCON has three major stages:

1. Topology selection
2. Forward performance prediction with a graph neural network
3. Layout-aware gradient reasoning over circuit parameters

The inverse-design step is:

```text
x* = argmin_x L_perf(f_theta(T, x), y_target) + lambda * L_layout(x)
```

where:

```text
T        = selected topology
x        = schematic/design parameters
f_theta  = learned GNN forward model
y_target = target performance vector
```

This assumes the optimization variables are differentiable and that the learned GNN provides useful gradients.

## Proposed Extension

Replace some analytic passive variables with generated pixelated EM blocks.

Instead of optimizing only:

```text
inductor radius
capacitor width
resistor length
```

generate actual passive geometry:

```text
metal pixels
vias
ports
keepouts
ground connections
```

The generator becomes:

```text
G_theta(target_specs, topology_context, port_constraints, stackup)
    -> layout logits
    -> hard pixelated passive
```

The hard passive is then evaluated by:

```text
EM solver
PNGF evaluator
surrogate ensemble
Spectre with generated Touchstone block
```

## Why ES Is Useful Here

The generated passive path contains non-differentiable steps:

```text
hard thresholding
DRC projection
EM meshing
solver failures
Touchstone generation
Spectre RF simulation
layout extraction
```

ES does not need gradients through any of those operations.

The ES training loop is:

```text
theta_i = theta + sigma * eps_i

layout_i = hard_threshold(G_theta_i(target_specs, topology_context))
layout_i = project_to_DRC_valid(layout_i)

S_passive_i = EM(layout_i)
perf_i = Spectre(active_circuit + S_passive_i)

fitness_i =
- circuit_spec_error(perf_i, target_specs)
- lambda_area * area(layout_i)
- lambda_drc * DRC_penalty(layout_i)
- lambda_loss * passive_loss(layout_i)

theta <- ES_update(theta, fitness_i)
```

This allows generator training against a black-box circuit/EM flow.

## Role In The FALCON Pipeline

The pixelated generator should not replace all of FALCON. It should augment the parts where geometry and EM behavior dominate.

Recommended hybrid:

```text
1. FALCON selects the circuit topology.
2. FALCON/GNN sizes active devices and simple passives.
3. Pixelated generator synthesizes difficult RF passives.
4. EM solver produces S-parameters for generated passives.
5. Spectre evaluates the full circuit with the passive n-port block.
6. ES updates the passive generator.
7. Final design is verified with full extraction / EM / Spectre.
```

So:

```text
FALCON = topology and active-device reasoning
Pixel generator = EM passive synthesis
ES = gradient-free training through black-box layout/EM/circuit flow
```

## Candidate Passive Blocks

High-value first targets:

1. Input matching networks for LNAs
2. Output matching networks for PAs
3. Interstage matching networks
4. Transformer/balun-like distributed passives
5. Couplers and dividers
6. Microstrip/CPW/SIW transitions
7. Parasitic compensation structures
8. Compact filters and notches

These blocks are often difficult to model with scalar parameters but natural to represent as pixelated layouts.

## Circuit-Level Scoring

There are two levels of scoring.

### Passive-Only EM Fitness

For a generated two-port or multiport passive:

```text
fitness =
- weighted_error(S_passive, S_target)
- area_penalty
- DRC_penalty
- insertion_loss_penalty
- isolation_penalty
```

This is useful when a desired passive S-parameter target is known.

### Full-Circuit Fitness

For a generated passive embedded in a circuit:

```text
fitness =
- gain_error
- bandwidth_error
- noise_figure_error
- output_power_error
- PAE_error
- return_loss_error
- stability_penalty
- area_penalty
- DRC_penalty
```

This is more expensive but directly optimizes the circuit-level objective.

## Example: LNA Input Matching

FALCON baseline:

```text
target specs -> choose LNA topology -> optimize transistor sizes and passive values
```

Pixelated extension:

```text
target specs
-> choose LNA topology
-> size active devices
-> generator creates input matching network layout
-> EM extracts matching-network S-parameters
-> Spectre evaluates NF, gain, S11, stability
-> ES improves generator
```

This can discover non-textbook matching geometries that trade off area, loss, and bandwidth.

## Example: PA Output Match

The output match strongly affects:

```text
output power
PAE
gain
bandwidth
harmonic behavior
stability
```

The generator can synthesize a distributed matching network whose EM response approximates the load-pull target.

Training objective:

```text
fitness =
- Pout_error
- PAE_error
- S22_error
- bandwidth_error
- instability_penalty
- area_penalty
```

## Recommended Architecture

The generator input should include:

```text
target circuit metrics
selected topology embedding
port locations
stackup/process metadata
allowed area/aspect ratio
frequency grid
optional desired passive S-parameters
latent noise vector
```

The output should be:

```text
metal logits [layers, height, width]
via logits [layer_pairs, height, width]
optional keepout/slot logits
```

Then:

```text
hard layout = project_to_valid_layout(threshold(logits))
```

The projection step should force:

```text
port pads
required grounds
keepout zones
minimum width/spacing
legal vias
connectivity constraints
```

## Training Strategy

Use staged training.

### Stage 1: Dataset Generation

Collect examples from:

```text
FALCON analytic passives
random valid pixelated passives
BPSO/CMA optimized passives
human-designed RF passives
```

Simulate them with:

```text
EM solver or PNGF
```

### Stage 2: Surrogate/Scorer Training

Train:

```text
layout -> passive S-parameters
layout + circuit context -> circuit metrics
```

Use an ensemble to estimate uncertainty.

### Stage 3: Generator Pretraining

Pretrain generator with:

```text
target passive response -> known/generated layout
```

or:

```text
target circuit context -> optimized passive layout
```

### Stage 4: ES Training Against Surrogate

Train cheaply:

```text
target -> generator -> hard layout -> surrogate -> fitness
```

### Stage 5: ES Fine-Tuning Against EM/Spectre

Use small populations:

```text
target -> generator -> hard layout -> EM -> Spectre -> fitness
```

Cache every solver result.

### Stage 6: Local Refinement

Use generator output as an initializer:

```text
generator layout -> BPSO/CMA/DBS pixel refinement -> final layout
```

Based on earlier antenna experiments, direct pixel optimization can outperform one-shot generation. The generator should therefore be judged both as a one-shot designer and as a proposal model.

## Connection To PNGF

The PNGF paper is highly relevant because ES and BPSO require many EM evaluations.

If a fixed passive design environment can be precomputed:

```text
fixed stackup
fixed ports
fixed optimization region
fixed frequency grid
```

then PNGF can replace slow EM calls:

```text
layout -> PNGF -> accurate S-parameters
```

This gives a strong scorer for:

```text
ES generator training
BPSO refinement
active learning data generation
surrogate dataset expansion
```

PNGF is especially valuable for local pixel changes because its low-rank update path is strongest when each optimization step modifies a small number of tiles.

## Comparison Experiments

For a fixed circuit topology, compare:

1. FALCON analytic passive sizing
2. Pixelated generator one-shot design
3. Pixelated generator plus BPSO refinement
4. Direct BPSO from random initialization
5. ES-trained generator against surrogate
6. ES-trained generator fine-tuned against EM/Spectre
7. PNGF-scored local optimization if PNGF is available

Primary metric:

```text
final extracted/cosimulated circuit performance under equal simulation budget
```

Secondary metrics:

```text
area
DRC validity
solver failures
number of EM calls
number of Spectre calls
time to usable design
robustness across target specs
```

## First Practical Experiment

Start with one fixed topology:

```text
LNA or PA
```

Keep active-device sizes fixed or initialized by FALCON.

Generate only one passive block:

```text
input match for LNA
or output match for PA
```

Use a two-port pixelated passive:

```text
port 1 = source/circuit side
port 2 = transistor/load side
```

Run:

```text
1. Generate passive layout.
2. EM simulate passive and export S2P.
3. Insert S2P into Spectre testbench.
4. Score full circuit metrics.
5. Train/refine with ES.
```

This isolates the passive-generator value without trying to automate the entire analog layout flow at once.

## Expected Outcome

The likely best result is not pure one-shot generation.

The most realistic winning flow is:

```text
FALCON topology + active sizing
-> pixelated passive generator
-> PNGF/EM local refinement
-> Spectre verification
```

This turns the generator into a fast proposal model and lets black-box optimization finish the geometry-level details.

## Key Takeaway

A pixelated ES-trained passive generator can extend FALCON from:

```text
differentiable schematic parameter inference
```

to:

```text
black-box EM-aware passive layout synthesis
```

That is especially valuable for mm-wave circuits, where passives, transitions, and interconnect geometry often dominate final performance.

