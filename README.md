# VLM-Powered Active Learning for Satellite Image Analysis

Adapting and evaluating an existing vision-language-model-based active prompt learning method for satellite image classification.

## Overview

This project explores active learning for satellite image classification using pretrained Vision-Language Models (VLMs). It builds on the research paper **Active Prompt Learning with Vision-Language Model Priors** (TMLR 2025) and its official implementation.

The project investigates the application of active prompt learning strategies to remote-sensing imagery, using EuroSAT experiments to study model adaptation and classification performance.

## Research Background

**Base paper:** Active Prompt Learning with Vision-Language Model Priors

- Paper: https://arxiv.org/abs/2411.16722
- Original implementation: https://github.com/kaist-dmlab/active-prompt-learning

The original method investigates budget-efficient active prompt learning through class-guided clustering, cluster-balanced sample acquisition, and selective querying using adaptive class-wise thresholds.

This project uses the existing implementation as its research foundation and focuses on adapting and evaluating the approach for satellite image classification.

## Objectives

- Explore active prompt learning for remote-sensing image classification.
- Experiment with VLM-based classification using the EuroSAT dataset.
- Run training and evaluation using different active-learning acquisition strategies.
- Analyze experimental results under different configurations and random seeds.

## Methodology

The experimental workflow consists of:

1. **Dataset preparation:** Configure the satellite image classification dataset.
2. **VLM-based representation:** Use the pretrained vision-language model supported by the implementation.
3. **Active sample selection:** Run the selected acquisition strategy to identify informative labeled examples.
4. **Prompt learning and training:** Execute experiments using the selected configuration.
5. **Evaluation:** Collect classification metrics from completed runs.
6. **Analysis:** Compare experimental outcomes across configurations where comparable results are available.

## Dataset

The experiments include EuroSAT, a satellite image dataset for land-use and land-cover classification.

Follow `DATASETS.md` for dataset preparation instructions. Dataset files are not included in this repository.

Dataset information: https://github.com/phelber/EuroSAT

## Repository Structure

- `configs/` — Experiment configurations.
- `scripts/` — Scripts for launching experiments.
- `trainers/` — Training and prompt-learning components.
- `clip/` — Vision-language model components.
- `dassl/` — Training framework components.
- `descriptors/` — Descriptor resources used by the implementation.
- `train.py` — Training entry point.
- `requirements.txt` — Python dependencies.
- `DATASETS.md` — Dataset setup instructions.

Refer to the source files for the exact implementation and available configurations.

## Installation

Create and activate a Python environment compatible with the original implementation:

```bash
conda create -n satellite-al python=3.10
conda activate satellite-al
pip install -r requirements.txt
```

Prepare the datasets according to `DATASETS.md`.

Check the original configuration and script paths before running experiments. Dataset paths and model settings must match your local environment.

## Running Experiments

The original implementation documents experiment launching through scripts under `scripts/alvlm/`.

For example, the original workflow uses a command of this form:

```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/alvlm/main.sh eurosat vit_b16 entropy 2 none
```

Verify that this script exists in your checkout and that its parameters match your intended experiment before running it. Update the GPU index, dataset path, model, acquisition strategy, seed, and mode as required.

## Experimental Results

Training and evaluation experiments have been conducted. Quantitative results will be documented here using measurements from completed runs.

For reproducibility, report:
- Model architecture and configuration.
- Active-learning acquisition strategy.
- Number or proportion of labeled training examples.
- Random seed and evaluation split.
- Classification metrics and experimental conditions.

Only include numerical comparisons supported by actual experiment outputs.

## Acknowledgements

This work builds on the original Active Prompt Learning with Vision-Language Model Priors implementation and its underlying codebase.

We acknowledge the authors and maintainers of the original project, CoOp, Dassl.pytorch, and the relevant dataset resources.

## Citation

If you use the original method, please cite the original paper:

```bibtex
@article{kim2025active,
  title={Active Prompt Learning with Vision-Language Model Priors},
  author={Kim, Hoyoung and Jin, Seokhee and Sung, Changhwan and Kim, Jaechang and Ok, Jungseul},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://arxiv.org/abs/2411.16722}
}
```

## License

See the included `LICENSE` file for the licensing terms of the original implementation. Review the original project's license and any dataset restrictions before redistributing modified code or associated assets.

## Project Status

Research adaptation and experimentation in progress.
