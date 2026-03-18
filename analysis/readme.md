## Overview

This repository contains custom Python code and a Minimal Working Example (MWE) to reproduce the findings of Onih et al. (2026). The analysis covers pupil-linked behavioral dynamics and hippocampal population geometry during unsupervised statistical learning.

## 1. System Requirements

### Hardware Requirements
- **RAM**: 16+ GB
- **Processor**: Quad-core 2.5GHz or faster

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 22.04+)
- **Python Version**: 3.9, 3.10, or 3.11
- **Core Dependencies**: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib, mne

## 2. Installation Guide (~5 minutes)

Clone the repository:
```bash
git clone https://github.com/LIMLabSWC/XdetectionCore.git
cd XdetectionCore
```

Set up environment:
```bash
python -m venv sl_env
source sl_env/bin/activate  # Windows: sl_env\Scripts\activate
pip install -e .
```

## 3. Test Data Generation
- **Generate Ephys Tensors**: `python gen_ephys_test_data.py` — Creates resps_by_cond dummy structures (Neurons x Time x Trials tensors)

## 4. Demos 

### A. Behavioral Pipeline (Figures 1 & 2)
Tests cluster-based permutation logic for significant pupil dilation responses:
```bash
python dummy_pupil_analysis.py
```
Output: Statistical clusters and PDR plots comparing 'Normal' vs 'Deviant' conditions.

### B. Neural Abstraction Pipeline (Figures 3, 4, & 5)
Tests population PCA (State-Space) and Rule Decoding (CCGP) logic:
```bash
python test_ephys_abstraction.py
```
Output: Verification of late-trial rule encoding and PCA trajectory separation.

## 6. License

This code is released under the MIT License.

## 7. Citation & Contact

**Citation**: Onih, A., et al. (2026). The hippocampus enables abstract structure learning without reward. 

**Contact**: Athena Akrami (athena.akrami@ucl.ac.uk)
