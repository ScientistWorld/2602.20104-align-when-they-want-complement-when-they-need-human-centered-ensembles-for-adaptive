# Adaptive Human-AI Collaboration

Official code for the AAAI 2026 paper:

> **Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration**    
> Hasan Amin, Ming Yin, and Rajiv Khanna  
> *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)*  
> [[Paper]](https://arxiv.org/abs/2602.20104)

## Overview

We propose an adaptive ensemble framework for human-AI collaboration that routes between alignment-optimized and complementarity-optimized AI specialists based on predicted human confidence. The method is grounded in a formal analysis of when alignment vs. complementarity is beneficial, and uses the CGPR team model for human-AI decision-making.

This repository contains code for both experimental settings in the paper:

1. **College Admission**: Synthetic 2D experiments that validate the theoretical results.
2. **WoofNette**: Image classification experiments with simulated human behavior (based on a real pilot study), training four DNN variants and two adaptive ensemble strategies.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

### College Admission

Runs all synthetic experiments and saves CSVs to `results/college_admission/`:

```bash
python college_admission.py
```

Individual experiments can be run with `--experiment`:

```bash
python college_admission.py --experiment specialist_divergence
```

Available experiments: `specialist_divergence`, `group_uncertainty`, `imperfect_human`, `group_proportion`, `weight_sweep`, `tradeoff`.

### WoofNette

**Data setup:** The WoofNette experiments require pre-processed image arrays and a trained routing proxy checkpoint. Full WoofNette data can be downloaded [here](https://drive.google.com/file/d/1ridMOHmfOp-vkNgwai5EinVcQQJlHDyY/view?usp=share_link). Set the paths in `config.yaml`:

```yaml
paths:
  data: /path/to/woofnette/data        # contains X_train.npy, X_test.npy, Y_train.npy, Y_test.npy
  models: /path/to/models              # for saving/loading model checkpoints
  woofnette_images: /path/to/images    # test images (for sample figure only)
```

The routing proxy can be trained with:

```bash
python confidence_predictor.py
```

Then set `woofnette.confidence_predictor.model_path` in `config.yaml` to the saved checkpoint path.

Run the full simulation across all seeds:

```bash
python woofnette.py
```

Or run specific seeds (0-indexed):

```bash
python woofnette.py --seeds 0 1
```

### Generate Figures

Regenerate all paper figures from saved results:

```bash
python analysis.py
```

Or specific figure sets:

```bash
python analysis.py --experiment ca        # College Admission figures
python analysis.py --experiment wn        # WoofNette figures
python analysis.py --experiment combined  # Combined 1x4 panel (key paper figure)
```

Figures are saved to `results/figures/`.

## Acknowledgements and Citation
This repository utilizes code and data from earlier work on [Behavior-Aware AI](https://github.com/shasanamin/ijcai24-behavior-aware-ai). Cite our papers using:

```bibtex
@inproceedings{amin2026align,
  title={Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration},
  author={Amin, Hasan and Yin, Ming and Khanna, Rajiv},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={21},
  pages={17337--17346},
  year={2026}
}
```

```bibtex
@inproceedings{mahmood2024designing,
  title={Designing behavior-aware AI to improve the human-AI team performance in AI-assisted decision making},
  author={Mahmood, Syed Hasan Amin and Lu, Zhuoran and Yin, Ming},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={3106--3114},
  year={2024}
}
```
