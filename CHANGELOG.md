# Changelog

All notable changes to the AI Pixel Antennas project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-10

### Initial Release

This is the first public release of the AI Pixel Antennas repository, accompanying the following publications:

- Gupta, A., Bhat, C., Karahan, E., Sengupta, K., & Khankhoje, U. (2023). "Tandem Neural Network based Design of Multi-band Antennas." *IEEE Transactions on Antennas and Propagation*, 71(8), 6308-6317.

- Gupta, A., & Khankhoje, U. (2024). "Transfer Learning Based Rapid Design of Frequency and Dielectric Agile Antennas." *IEEE Journal on Multiscale and Multiphysics Computational Techniques*, 10, 47-57.

### Added

#### MATLAB Code
- `generateantenna_air.m` - Generate single air-substrate antenna structure
- `generateantenna_for_tandem_air.m` - Parallel dataset generation for air substrates
- `generateantenna_scaled.m` - Generate scaled dielectric-substrate antenna
- `generateantenna_transferlearning.m` - Dataset generation for transfer learning
- `inverse_design_using_bpso_with_TLfwdmodel.m` - Binary PSO inverse design with neural surrogate
- `parsave.m` - Helper function for parallel dataset saving

#### Python/Jupyter Notebooks
- `Inverse_design_tandem.ipynb` - Training code for tandem neural network
- `Test_Inverse_design_tandem.ipynb` - Inference code for antenna design generation
- `Forward_model_Transfer_learning.ipynb` - Transfer learning implementation

#### Pre-trained Models (External Downloads)
- Forward surrogate CNN for tandem network (air, 10-20 GHz)
- Inverse network for tandem architecture
- Transfer learning forward surrogate (FR-4, 1-5 GHz)

#### Datasets (External Downloads)
- Air-substrate antenna database (500k samples)

#### Documentation
- `README.md` - Comprehensive project documentation
- `LICENSE` - MIT License (No Patent Grant)
- `PATENT_NOTICE` - Patent licensing information
- `CHANGELOG.md` - This file

#### Figures
- `tandem_overview.png` - Conceptual framework comparison
- `tandem_architecture.png` - Network architecture diagram
- `forward_cnn.png` - Forward surrogate CNN structure
- `transfer_learning_flow.png` - Transfer learning workflow
- `pixelated_patch.png` - Pixelated antenna representation
- `single_band_result.png` - Example design result

### Repository Information

- **License:** MIT (No Patent Grant) - See LICENSE file
- **Patent:** Indian Patent No. 572928 - See PATENT_NOTICE file
- **Authors:** Aggraj Gupta, Uday Khankhoje
- **Affiliation:** Department of Electrical Engineering, IIT Madras

---

## Future Releases

Future versions may include:
- Additional substrate materials and frequency bands
- Extended pixel grid resolutions
- Pre-trained models for different antenna configurations
- Jupyter notebook tutorials for beginners
- Validation scripts for comparing surrogate vs. full-wave simulations
- Performance benchmarking tools

---

For questions or contributions, please contact the authors or open an issue in the repository.
