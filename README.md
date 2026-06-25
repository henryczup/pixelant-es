# AI Pixel Antennas

[![License](https://img.shields.io/badge/License-MIT%20(no%20patent%20grant)-blue.svg)](LICENSE)
[![Patent](https://img.shields.io/badge/Patent-Indian%20Patent%20572928-red.svg)](PATENT_NOTICE)

Deep learning-enabled inverse design of compact pixelated antennas using tandem neural networks, transfer learning, and or evolutionary algorithms.

---

## Authors

**Aggraj Gupta** and **Uday Khankhoje**
Department of Electrical Engineering
Indian Institute of Technology Madras

**Contact:**
- Aggraj Gupta: [gupta.aggraj@gmail.com](mailto:gupta.aggraj@gmail.com)
- Uday Khankhoje: [uday@ee.iitm.ac.in](mailto:uday@ee.iitm.ac.in)

---

## Overview

This repository provides complete workflows for neural network-based inverse design of pixelated microstrip antennas. The methods enable rapid synthesis of compact single-band and multi-band antennas from desired electromagnetic specifications through either a tandem neural network architecture or a neural-network backed evolutionary optimization approach. The key work has ben published in the IEEE Transactions on Antennas & Propagation and the IEEE Journal on Multiscale and Multiphysics Computational Techniques (see detailed information below).

### Key Features

- **Tandem Neural Network Architecture**: Combines inverse and forward surrogate models to resolve non-uniqueness in inverse design
- **Binary Pixelated Parameterization**: 12×12 pixel grid representation enabling exploration of 2^142 possible designs
- **Transfer Learning Framework**: Reduces required simulations by 88% when adapting from air to dielectric substrates
- **Multiple Design Methods**: Supports both neural network-based and evolutionary (BPSO) optimization approaches
- **Rapid Design Generation**: Sub-second antenna synthesis compared to hours of traditional optimization

![Tandem Overview](tandem_overview.png)

*Traditional trial-and-error vs. tandem neural network-based inverse synthesis*

---

## Publications

This code implements methods described in the following peer-reviewed publications:

### 1. Base Paper: Tandem Neural Network Design

**"Tandem Neural Network based Design of Multi-band Antennas"**
Aggraj Gupta, Chandan Bhat, Emir Karahan, Kaushik Sengupta, Uday Khankhoje
*IEEE Transactions on Antennas and Propagation*, vol. 71, no. 8, pp. 6308-6317, 2023
DOI: [10.1109/TAP.2023.3276524](https://doi.org/10.1109/TAP.2023.3276524)

**Key Contributions:**
- Tandem architecture combining inverse network with frozen forward surrogate
- Smooth thresholding activation for binary design enforcement
- Joint loss function balancing spectrum fidelity, design consistency, and binary regularization

### 2. Transfer Learning Extension

**"Transfer Learning Based Rapid Design of Frequency and Dielectric Agile Antennas"**
Aggraj Gupta, Uday Khankhoje
*IEEE Journal on Multiscale and Multiphysics Computational Techniques*, vol. 10, pp. 47-57, 2024
DOI: [10.1109/JMMCT.2024.3509773](https://doi.org/10.1109/JMMCT.2024.3509773)

**Key Contributions:**
- Transfer learning strategy leveraging fast air-substrate simulations
- Scaling laws for frequency and dielectric migration
- 88% reduction in required dielectric simulations (from 500k to 60k samples)

### Related Work: Multi-Port RF Systems

**"Deep-learning Enabled Generalized Inverse Design of Multi-Port Radio-frequency and Sub-Terahertz Passives and Integrated Circuits"**
Emir Karahan, Zheng Liu, Aggraj Gupta, Zijian Shao, Jonathan Zhou, Uday Khankhoje, Kaushik Sengupta
*Nature Communications*, vol. 15, article 10734, 2024
DOI: [10.1038/s41467-024-54178-1](https://doi.org/10.1038/s41467-024-54178-1)

This publication demonstrates the broader applicability of the surrogate-based inverse design paradigm to filters, couplers, and impedance matching networks.

---

## Patent Information

**IMPORTANT:** The algorithms implemented in this code are protected by patent. See [PATENT_NOTICE](PATENT_NOTICE) for complete details.

**Patent:** Method Of Designing Ultra Compact Single Band Antennas
**Inventors:** Uday Khankhoje and Aggraj Gupta
**Indian Patent No. 572928** (Filed: 22 Jan 2024, Granted: 30 Oct 2025)

- **Academic/Research Use:** Free with proper attribution (cite papers above)
- **Commercial Use:** Requires separate patent license (contact authors)

---

## Technical Overview

### Pixelated Antenna Model

![Pixelated Antenna](pixelated_patch.png)

- Base patch tessellated into **12×12 binary pixels**
- Each pixel represents metal (1) or no-metal (0)
- Feed-adjacent pixels fixed to ensure connectivity
- Avoids template bias, enables non-intuitive geometries

### Tandem Neural Network Architecture

![Tandem Architecture](tandem_architecture.png)

The tandem architecture consists of:

1. **Inverse Network**: Maps desired S₁₁(f) spectrum → pixelated geometry
2. **Forward CNN Surrogate**: Maps geometry → S₁₁(f) (frozen weights)
3. **Joint Training**: Ensures data consistency despite non-unique inverse mapping

**Loss Function:**
```
L = L_S + α·L_D + β·L_B
```
- L_S: Spectrum prediction error
- L_D: Design consistency error
- L_B: Binary regularization term

**Smooth Thresholding Activation:**
```
f(x) = 0.5 + 0.5·tanh(m(x - 0.5))
```
Enforces binary outputs during training, not as post-processing.

### Transfer Learning Strategy

![Transfer Learning Flow](transfer_learning_flow.png)

**Motivation:** Air-filled simulations are 50-60× faster than dielectric simulations.

**Workflow:**
1. Train forward CNN on 500k air-filled antennas (10-20 GHz)
2. Apply scaling laws to map air designs to dielectric domain
3. Fine-tune with 60k dielectric antennas (FR-4, 1-5 GHz)
4. Achieve 88% reduction in simulation cost

**Scaling Law:**
```
w_d = a·w_a,  where  a = f_a / (f_d·√ε_eff)
```
Typical scale factors: 2-5 for migration from 10-20 GHz (air) to 1-5 GHz (FR-4).

---

## Repository Contents

### MATLAB Code Files

| File | Function Signature | Description |
|------|-------------------|-------------|
| `generateantenna_air.m` | `p = generateantenna_air(x_dis, y_dis, ant_des)` | Generate single air-substrate antenna structure |
| `generateantenna_for_tandem_air.m` | `generateantenna_for_tandem_air(x_dis, y_dis, samples_to_generate)` | Generate dataset of air-substrate antennas in parallel |
| `generateantenna_scaled.m` | `p = generateantenna_scaled(x_dis, y_dis, ant_des)` | Generate single dielectric-substrate antenna with scaling |
| `generateantenna_transferlearning.m` | `generateantenna_transferlearning(x_dis, y_dis, samples_to_generate)` | Generate dielectric antenna dataset for transfer learning |
| `inverse_design_using_bpso_with_TLfwdmodel.m` | Script (no function) | Evolutionary inverse design using Binary PSO with neural surrogate |
| `parsave.m` | `parsave(fname, Test_patches, spec)` | Helper function for parallel dataset saving |

**Function Parameters:**
- `x_dis`, `y_dis`: Grid discretization (number of pixels + 1, typically 13)
- `ant_des`: 12×12 binary matrix specifying antenna geometry
- `samples_to_generate`: Number of antenna samples to simulate
- `fname`: Output filename for saved data
- `Test_patches`: Flattened antenna design vector (144 elements)
- `spec`: S₁₁ spectrum (81 frequency points)

**Returns:**
- `p`: MATLAB pcbStack object representing complete antenna structure

### Python/Jupyter Notebooks

| File | Description |
|------|-------------|
| `Inverse_design_tandem.ipynb` | Training code for tandem neural network (forward + inverse) |
| `Test_Inverse_design_tandem.ipynb` | Inference code to generate antenna designs from target spectra |
| `Forward_model_Transfer_learning.ipynb` | Transfer learning implementation for dielectric substrates |

### Pre-trained Models

**Note:** Due to file size limitations, pre-trained models are hosted externally. Download links:

| File | Description | Size | Download Link |
|------|-------------|------|---------------|
| `Forward_model_for_tandem.pth` | Forward surrogate CNN for tandem network (air, 10-20 GHz) | 512 MB | [Download](https://www.dropbox.com/scl/fi/jx5dziz6112n2rv51vkji/Forward_model_for_tandem.pth?rlkey=i9elgksepyk175lgldm9vwm49&dl=0) |
| `inverse_tandem_model.pth` | Inverse network for tandem architecture | 63 MB | [Download](https://www.dropbox.com/scl/fi/qmhk9n8t4tk8awrt2hhwh/inverse_tandem_model.pth?rlkey=smw944xa8yzelnk5jr30xf1bo&dl=0) |
| `TLfwdmodel` | Transfer learning forward surrogate (FR-4, 1-5 GHz) | 171 MB | [Download](https://www.dropbox.com/scl/fi/vzrog9tmdewgd4og79lue/TLfwdmodel?rlkey=edx0rb8r6j8ksq0ncfc4s8s2e&dl=0) |

After downloading, place these files in the root directory of the repository.

### Datasets

**Note:** Due to file size limitations, datasets are hosted externally. Download links:

| File | Description | Size | Download Link |
|------|-------------|------|---------------|
| `antenna_dataset.mat` | Air-substrate antenna database (500k samples) | 436 MB | [Download](https://www.dropbox.com/scl/fi/it2ru3gnct3c9q9px6xv5/antenna_dataset.mat?rlkey=mfv1vj4i30cgor5x1bnt7xf6e&dl=0) |

After downloading, place this file in the root directory of the repository.

### Figures

High-resolution images illustrating the methodology:
- `tandem_overview.png` - Conceptual comparison: traditional vs. AI-based design
- `tandem_architecture.png` - Detailed tandem network architecture diagram
- `forward_cnn.png` - Forward surrogate CNN structure
- `transfer_learning_flow.png` - Transfer learning workflow
- `pixelated_patch.png` - Pixelated antenna representation
- `single_band_result.png` - Example single-band antenna design result

---

## Usage Guide

### Prerequisites

**MATLAB Requirements:**
- MATLAB R2020b or later
- Antenna Toolbox
- Parallel Computing Toolbox (for dataset generation)

**Python Requirements:**
- Python 3.8+
- PyTorch 1.10+
- NumPy, Matplotlib
- Jupyter Notebook

### Use Case 1: Generate Single Air-Substrate Antenna

Create and visualize a single antenna structure on air substrate:

```matlab
% Define a 12×12 binary antenna design
ant_design = randi([0,1], 12, 12);
ant_design(6:7, 1) = 1;  % Ensure feed connectivity

% Generate antenna structure (13 = 12 pixels + 1)
antenna = generateantenna_air(13, 13, ant_design);

% Simulate S-parameters
freq = linspace(10e9, 20e9, 81);
s = sparameters(antenna, freq, 50);
rfplot(s, 1, 1);
```

**Inputs:**
- `13, 13`: Grid dimensions (12×12 pixels requires 13 grid points)
- `ant_design`: 12×12 binary matrix (1=metal, 0=no metal)

**Outputs:**
- `antenna`: pcbStack object with air substrate, ready for EM simulation

### Use Case 2: Generate Air-Substrate Dataset

Create a large dataset for neural network training:

```matlab
% Generate 1000 random air-substrate antennas
generateantenna_for_tandem_air(13, 13, 1000);

% Output: Individual .mat files (output1.mat, output2.mat, ...)
% Each contains: Test_patches (144×1 design vector), spec (81×1 S11 spectrum)
```

**Note:** Uses 8 parallel workers. Adjust `parpool('local', 8)` based on available cores.

### Use Case 3: Transfer Learning for Dielectric Substrates

Scale air-substrate designs to FR-4 and generate dielectric dataset:

```matlab
% First, create scaled dielectric design
ant_design = randi([0,1], 12, 12);
ant_design(6:7, 1) = 1;

% Generate FR-4 antenna (scaling factor a=4 built-in)
antenna_fr4 = generateantenna_scaled(13, 13, ant_design);

% Generate full dataset for transfer learning
generateantenna_transferlearning(13, 13, 5000);
```

**Substrate Parameters (built-in):**
- Material: FR-4
- ε_r = 4.8, tan δ = 0.026
- Thickness: 3.2 mm
- Frequency range: 1-5 GHz

### Use Case 4: Inverse Design Using Tandem Neural Network

Generate antenna from desired spectrum using pre-trained model:

```matlab
% See Test_Inverse_design_tandem.ipynb for complete workflow
```

**Python workflow:**
1. Load pre-trained models (`Forward_model_for_tandem.pth`, `inverse_tandem_model.pth`)
2. Define target S₁₁(f) spectrum (81 frequency points)
3. Run inverse network to generate 12×12 design
4. Validate using forward surrogate
5. Export to MATLAB for final EM verification

### Use Case 5: Evolutionary Inverse Design (BPSO)

Use Binary Particle Swarm Optimization with neural surrogate:

```matlab
% Open and configure inverse_design_using_bpso_with_TLfwdmodel.m
% Set target frequency (line 11):
center_fiu = find(freq == 3.6e9);  % Target: 3.6 GHz

% Set band type (lines 20-26):
% - Single band: pass_band = [center_fiu-4:center_fiu+4]
% - Dual band: uncomment dual band section

% Run script (outputs best design and convergence plot)
run('inverse_design_using_bpso_with_TLfwdmodel.m')
```

**Algorithm Parameters:**
- Population size: 1000
- Max iterations: 50
- Inertia weight: 0.9 → 0.4 (linearly decreasing)
- Acceleration factors: c1=c2=2

**Output:**
- `antenna_des`: Optimized 12×12 binary design
- `output_new`: Predicted S₁₁ spectrum
- Convergence plot and S₁₁ response plot

### Use Case 6: EGGROLL Direct Antenna Search

This path replaces direct GA/BPSO mask evolution with EGGROLL over a small neural generator:

```text
target spectrum + latent noise -> generator -> mask logits -> hard threshold -> scorer -> fitness
```

For the original 12x12 antenna case, this remains compatible with the frozen forward surrogate and optional ST-trained inverse warm start. For larger layouts such as 32x32, 64x64, multilayer metasurfaces, or future 3D encodings, use the CNN generator and provide a compatible scorer. This is the more compelling regime because the generator parameterizes a structured layout distribution while GA/BPSO must search the full binary mask space directly.

Two scorer families are implemented:
- `surrogate`: the frozen PyTorch/JAX forward surrogate.
- `external-em`: a cached MATLAB Antenna Toolbox scorer for 12x12 hard masks.

The external EM scorer has two modes:
- `air`: faster exploratory solve, default 10-20 GHz grid.
- `substrate`: slower FR-4 solve using the `generateantenna_scaled.m` defaults, default 1-5 GHz grid.

The implementation uses the official HyperscaleES EGGROLL implementation as an external dependency, not vendored into this repository.

Install the optional dependencies:

```bash
pip install -r requirements-eggroll.txt
```

Run a warm-started EGGROLL phase from an ST-trained inverse checkpoint:

```bash
python train_eggroll_inverse.py \
  --spectra-mat antenna_dataset.mat \
  --forward-checkpoint Forward_model_for_tandem.pth \
  --inverse-checkpoint inverse_tandem_model.pth \
  --output-dir eggroll_runs/warm_start \
  --layout 12x12 \
  --generator mlp \
  --population-size 256 \
  --steps 100 \
  --rank 1 \
  --lr 1e-3 \
  --lambda-conn 0.1 \
  --lambda-area 0.01 \
  --lambda-frag 0.01 \
  --lambda-rule 0.01
```

Run from scratch by omitting `--inverse-checkpoint`:

```bash
python train_eggroll_inverse.py \
  --spectra-mat antenna_dataset.mat \
  --forward-checkpoint Forward_model_for_tandem.pth \
  --output-dir eggroll_runs/scratch \
  --layout 12x12 \
  --generator mlp \
  --population-size 512 \
  --steps 200 \
  --rank 1 \
  --sigma 0.2
```

Run the scalable direct-search path for a larger layout once a compatible scorer is available:

```bash
python train_eggroll_inverse.py \
  --spectra-mat antenna_dataset.mat \
  --forward-checkpoint Forward_model_32x32.pth \
  --output-dir eggroll_runs/direct_32x32 \
  --layout 32x32 \
  --generator cnn \
  --latent-dim 16 \
  --population-size 512 \
  --steps 200 \
  --rank 1
```

Key generalized options:
- `--layout HxW` selects the generated mask size, for example `12x12`, `32x32`, or `64x64`.
- `--layers N` emits multilayer logits shaped `[batch, layers, height, width]`.
- `--generator mlp|cnn` selects the legacy fully connected generator or spatial CNN generator.
- `--latent-dim N` appends per-sample latent noise to the target spectrum before generation.
- `--scorer surrogate|external-em` selects the frozen surrogate adapter or the future full-wave adapter contract.
- `--solver-mode air|substrate` selects the MATLAB EM solver when `--scorer external-em` is used.
- `--scorer-cache-dir PATH` caches MATLAB spectra by binary mask and solver configuration.

Run hard-binary EGGROLL against the fast air EM scorer:

```bash
python train_eggroll_inverse.py \
  --spectra-mat antenna_dataset.mat \
  --scorer external-em \
  --solver-mode air \
  --scorer-work-dir . \
  --scorer-cache-dir eggroll_runs/em_cache_air \
  --output-dir eggroll_runs/eggroll_air_em \
  --layout 12x12 \
  --generator mlp \
  --hidden-dims 64,64 \
  --population-size 8 \
  --steps 5 \
  --rank 1 \
  --sigma 0.1
```

Switch to the slower FR-4/substrate scorer by changing `--solver-mode substrate` and using a separate cache directory. Keep the initial population and step count small; each uncached population member launches a full-wave MATLAB solve.

You can also call the MATLAB batch scorer directly:

```matlab
score_designs_em('input.mat', 'output.mat')
```

where `input.mat` contains `designs`, optional `freq`, and `solver_mode`; `output.mat` contains `s11_db`, `valid`, `error_message`, and `elapsed_seconds`.

**Implementation details:**
- Generator output is standardized as logits shaped `[batch, layers, height, width]` and converted with `(logits > 0)`; the smooth threshold activation is not used in this phase.
- Feed-adjacent MATLAB pixels `X(6:7, 1)` are forced to metal after thresholding for the default 12x12 layout.
- Antithetic perturbations use paired population members, so `--population-size` must be even.
- The trainer writes `metrics.csv`, `best_design.png`, and portable inverse-generator `.npz` checkpoints under `--output-dir`.

---

## Example Results

### Single-Band Antenna (3.6 GHz, FR-4)

![Single Band Result](single_band_result.png)

**Performance Metrics:**
- Resonant frequency: 3.6 GHz
- Return loss: < -20 dB
- Fractional bandwidth: ~5%
- Gain: ~3.5 dBi
- Radiation efficiency: > 70%

### Multi-Band Capabilities

The tandem network can synthesize dual-band and triple-band antennas with:
- Independent control of resonant frequencies
- Up to 50% area reduction vs. conventional patches
- Compact, non-intuitive geometries

---

## Important Notes and Limitations

### Scope of Implementation

**The code provided in this repository covers specific use cases:**
- 12×12 pixelated microstrip antennas
- Air and FR-4 substrates
- Frequency ranges: 10-20 GHz (air), 1-5 GHz (FR-4)
- Single-band and limited multi-band designs

**The algorithms in the published papers can be instantiated in many other settings NOT covered in this repository, including:**
- Different pixel grid resolutions (e.g., 16×16, 24×24)
- Alternative substrate materials (Rogers, alumina, etc.)
- Different frequency bands (sub-GHz, mmWave, sub-THz)
- Other antenna types (slots, monopoles, arrays)
- Different EM objectives (gain, efficiency, radiation patterns)
- Multi-port networks and RF passives (see Nature Comm. paper)

### Computational Requirements

- **Dataset generation:** Computationally intensive; 500k air antennas ≈ weeks on 8-core workstation
- **Pre-trained models provided:** Use these to avoid re-training
- **Neural network training:** Requires GPU (8+ GB VRAM recommended)
- **Inference:** Fast on CPU (< 1 second per design)

### Validation

**All neural network predictions should be validated with full-wave EM simulation before fabrication.** The surrogate models provide excellent approximations but may have errors for edge cases or out-of-distribution designs.

---

## How to Cite

If you use this code or methodology in your research, please cite the relevant publications:

### For Tandem Neural Network Method:

```bibtex
@article{gupta2023tandem,
  title={Tandem Neural Network Based Design of Multi-band Antennas},
  author={Gupta, Aggraj and Bhat, Chandan and Karahan, Emir and Sengupta, Kaushik and Khankhoje, Uday},
  journal={IEEE Transactions on Antennas and Propagation},
  volume={71},
  number={8},
  pages={6308--6317},
  year={2023},
  doi={10.1109/TAP.2023.3276524}
}
```

### For Transfer Learning Method:

```bibtex
@article{gupta2024transfer,
  title={Transfer Learning Based Rapid Design of Frequency and Dielectric Agile Antennas},
  author={Gupta, Aggraj and Khankhoje, Uday},
  journal={IEEE Journal on Multiscale and Multiphysics Computational Techniques},
  volume={10},
  pages={47--57},
  year={2024},
  doi={10.1109/JMMCT.2024.3509773}
}
```

### For Related Multi-Port RF Systems:

```bibtex
@article{karahan2024deep,
  title={Deep-learning Enabled Generalized Inverse Design of Multi-Port Radio-frequency and Sub-Terahertz Passives and Integrated Circuits},
  author={Karahan, Emir Ali and Liu, Zheng and Gupta, Aggraj and Shao, Zijian and Zhou, Jonathan and Khankhoje, Uday and Sengupta, Kaushik},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={10734},
  year={2024},
  doi={10.1038/s41467-024-54178-1}
}
```

---

## License and Patent Information

### Copyright License

This software is licensed under the **MIT License (No Patent Grant)**. See [LICENSE](LICENSE) file for complete terms.

**In brief:** You may freely use, modify, and distribute the source code, but WITHOUT any patent rights.

### Patent Notice

**The copyright license does NOT include any patent license.** The algorithms implemented in this code are protected by **Indian Patent No. 572928**.

- **Academic/Non-Commercial Use:** Encouraged with proper citation
- **Commercial/Industrial Use:** Requires separate patent license

See [PATENT_NOTICE](PATENT_NOTICE) for complete details and contact information for commercial licensing.

---

## Acknowledgments

This work was supported by research funding at IIT Madras and a research grant “6G: Sub-THz Wireless Communication with Intelligent Reflecting Surfaces (IRS)” 
numbered R‐23011/3/2022‐CC&BT‐MeitY by the Ministry of Electronics and Information Technology (MeitY), Government of India. We thank our collaborators at Princeton University (Prof. Kaushik Sengupta's group) for contributions to the foundational research.

---

## Questions and Support

For technical questions about the code:
- Open an issue in this repository (when public)
- Contact: [gupta.aggraj@gmail.com](mailto:gupta.aggraj@gmail.com)

For patent licensing inquiries:
- Contact: [uday@ee.iitm.ac.in](mailto:uday@ee.iitm.ac.in)

For academic collaborations:
- Contact either author via emails above

---

**Last Updated:** February 2026
