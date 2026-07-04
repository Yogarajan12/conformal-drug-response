# Uncertainty-Aware Cancer Drug Response Prediction

Predicting drug sensitivity (log IC50) for drug–cell-line pairs on **GDSC1** with a Graph Neural Network, and wrapping the predictions in **split conformal prediction** to obtain distribution-free coverage guarantees. An MC-dropout baseline is included to show why calibrated intervals matter.

---

## Overview

Each sample is a (drug, cell line) pair. The drug is encoded from its molecular graph with a GCN; the cell line is encoded from its 17,737-gene expression profile with an MLP. The two embeddings are concatenated and passed to a regression head that predicts log(IC50).

On top of the point predictor, split conformal prediction calibrates prediction intervals on a held-out set so that empirical coverage matches the target level `1 − α` — with a finite-sample validity guarantee that holds regardless of the model or data distribution. The same setup is evaluated against MC dropout, which turns out to be badly miscalibrated here, motivating the conformal approach.

## Key results (test set, n = 26,597)

| Metric | Value |
|---|---|
| RMSE | 1.184 |
| MAE | 0.876 |
| Pearson r | 0.909 |
| Spearman ρ | 0.889 |
| Coverage (90% target) | 90.2% |
| Mean interval width | 3.795 |

Coverage holds tightly across every target level tested:

| Target coverage (1 − α) | Empirical coverage | Mean interval width |
|---|---|---|
| 80% | 80.2% | 2.804 |
| 85% | 85.1% | 3.233 |
| 90% | 90.1% | 3.852 |
| 95% | 95.0% | 4.885 |
| 98% | 97.8% | 6.283 |

A binomial test on the 90% intervals gives p = 0.18 — empirical coverage is statistically indistinguishable from the 90% target.

**Why conformal over MC dropout:** with 50 stochastic forward passes and 90% nominal intervals, MC dropout achieved only ~48.6% coverage and a weak uncertainty–error correlation (Spearman ≈ 0.11). Split conformal recovers the guarantee that MC dropout misses.

## Method

**Drug encoder.** 3-layer GCN with residual connections and batch norm over a 69-dimensional atom featurization (atom type, degree, formal charge, hybridization, aromaticity, hydrogen count). Graph-level readout combines mean and max pooling.

**Cell-line encoder.** MLP over standardized gene-expression vectors (17,737 genes → dense embedding).

**Head.** Concatenated drug and cell embeddings → MLP → scalar log(IC50). ~4.71M parameters total.

**Conformal layer.** Split conformal prediction with absolute-residual nonconformity scores; the interval half-width is the finite-sample-corrected empirical quantile of calibration residuals. The calibration split is kept strictly separate from training so the coverage guarantee remains valid.

**Data split.** 70% train / 15% calibration / 15% test (124,117 / 26,596 / 26,597 samples; 208 drugs, 958 cell lines).

## Repository structure

```
.
├── notebooks/
│   └── drug_response_conformal.ipynb   # main notebook (add your Deepnote export here)
├── figures/                            # generated plots
├── results/                            # metrics, predictions, coverage tables, LaTeX
│   ├── test_metrics.csv
│   ├── coverage_analysis.csv
│   ├── predictions.csv
│   ├── experiment_summary.json
│   └── results_table.tex
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Large binaries — the cached GDSC pickle (`data/gdsc1.pkl`) and model checkpoints (`*.pt`) — are intentionally not tracked (see `.gitignore`); GDSC1 is re-downloaded automatically by TDC.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install `torch` and `torch-geometric` matching your CUDA / CPU build first if the default wheels don't resolve — see the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) install guides.

## Usage

Open the notebook and run top to bottom:

```bash
jupyter lab notebooks/drug_response_conformal.ipynb
```

GDSC1 downloads automatically via Therapeutics Data Commons on first run. The pipeline handles featurization, training with early stopping, conformal calibration across α levels, the MC-dropout comparison, and figure/metric export.

## Dataset

GDSC1 (Genomics of Drug Sensitivity in Cancer) via [Therapeutics Data Commons](https://tdcommons.ai/). 177,310 drug–cell-line measurements, 208 drugs, 958 cell lines, log(IC50) targets.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Author

Yogarajan — interpretable and uncertainty-aware ML for clinical and biomedical applications.
