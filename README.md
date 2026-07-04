# Uncertainty-Aware Cancer Drug Response Prediction

A graph neural network that predicts drug sensitivity (log IC50) for drug and cell-line pairs on **GDSC1**, wrapped in **split conformal prediction** to produce prediction intervals with a distribution-free coverage guarantee. An MC-dropout baseline is included to show why calibrated intervals matter in practice.

---

## Motivation

Most drug response models report a single accuracy number and stop there. For any use downstream of the model, a point estimate of log(IC50) is not enough: a prediction of "this drug is effective against this cell line" is only actionable if it comes with a trustworthy sense of how uncertain that prediction is. A model that is confidently wrong is worse than one that flags its own unreliability.

The difficulty is that the usual ways of getting uncertainty out of a neural network (softmax scores, MC dropout, deep ensembles) give heuristic estimates with no formal guarantee that they are calibrated. This project takes a different route. It keeps a standard, well-performing GNN regressor as the point predictor and adds a thin conformal layer on top that turns the model's residuals into intervals with a provable, finite-sample coverage guarantee. The guarantee holds regardless of the model architecture or the data distribution, and requires only that the calibration and test data are exchangeable.

## Overview

Each sample is a (drug, cell line) pair with a continuous log(IC50) target. The pipeline has three parts:

1. **Point predictor.** A GNN encodes the drug from its molecular graph and an MLP encodes the cell line from its gene-expression profile. The two embeddings are concatenated and passed to a regression head.
2. **Conformal calibration.** A held-out calibration split is used to compute nonconformity scores (absolute residuals) and derive an interval half-width that achieves the target coverage level, with a finite-sample correction to the empirical quantile.
3. **Uncertainty comparison.** The conformal intervals are benchmarked against MC dropout to make the calibration difference concrete.

## Key results (test set, n = 26,597)

| Metric | Value |
|---|---|
| RMSE | 1.184 |
| MAE | 0.876 |
| Pearson r | 0.909 |
| Spearman rho | 0.889 |
| Coverage (90% target) | 90.2% |
| Mean interval width | 3.795 |

The point predictor reaches a Pearson correlation of 0.909 between predicted and measured log(IC50), which is competitive for GDSC1 regression. More importantly, the conformal intervals are tightly calibrated: at every target level tested, empirical coverage lands within a fraction of a percent of nominal.

| Target coverage (1 - alpha) | Empirical coverage | Mean interval width |
|---|---|---|
| 80% | 80.2% | 2.804 |
| 85% | 85.1% | 3.233 |
| 90% | 90.1% | 3.852 |
| 95% | 95.0% | 4.885 |
| 98% | 97.8% | 6.283 |

A two-sided binomial test on the 90% intervals gives p = 0.18, so the observed coverage is statistically indistinguishable from the 90% target. Interval width grows smoothly with the confidence level, which is the expected accuracy-versus-certainty trade-off: higher confidence costs wider intervals.

### Why conformal over MC dropout

Running MC dropout with 50 stochastic forward passes and forming 90% intervals as mean plus or minus 1.645 standard deviations yields only about 48.6% empirical coverage, roughly half the nominal target, and a weak rank correlation between the estimated uncertainty and the actual prediction error (Spearman rho around 0.11). In other words, MC dropout here is both badly miscalibrated and a poor guide to where the model is likely to be wrong. Split conformal recovers the coverage guarantee that MC dropout misses, at a fraction of the inference cost (one forward pass instead of 50).

### Conditional coverage

Coverage is not perfectly uniform across the prediction range. Binning test predictions into five groups shows coverage running from roughly 85% in the lowest-prediction bins up to about 95% in the highest. This is a known limitation of vanilla split conformal, which guarantees marginal coverage but not conditional coverage. It is called out here honestly rather than hidden, and it points naturally toward locally adaptive or normalized conformal variants as a next step.

## Method

**Molecular featurization.** Each drug SMILES string is parsed with RDKit into a molecular graph. Every atom is described by a 69-dimensional feature vector: a one-hot atom-type encoding over 44 elements, one-hot degree, one-hot formal charge, one-hot hybridization state, an aromaticity flag, and a normalized hydrogen count. Bonds become undirected edges.

**Drug encoder.** A 3-layer graph convolutional network with residual connections and batch normalization operates on the atom features. Node embeddings are aggregated into a single graph-level vector by summing mean pooling and max pooling, which captures both average and salient substructure signals.

**Cell-line encoder.** Gene-expression vectors (17,737 genes per cell line) are standardized and passed through a multilayer perceptron that compresses them to a dense embedding of the same dimension as the drug embedding.

**Prediction head.** The drug and cell embeddings are concatenated and fed to an MLP that outputs a scalar log(IC50). The full model has roughly 4.71M parameters.

**Training.** Mean squared error loss, Adam optimizer at a 1e-3 learning rate, ReduceLROnPlateau scheduling, gradient clipping at norm 1.0, and early stopping on the calibration split with a patience of 15 epochs. Training converged and stopped early around epoch 45.

**Conformal layer.** Split conformal prediction with absolute-residual nonconformity scores. The interval half-width q is the finite-sample-corrected empirical quantile of the calibration residuals at level ceil((n + 1)(1 - alpha)) / n. Intervals are then formed as prediction plus or minus q. Because q is computed only on the calibration split, which is disjoint from training, the coverage guarantee remains valid.

**Data split.** A 70 / 15 / 15 partition into training, calibration, and test sets, giving 124,117 / 26,596 / 26,597 samples across 208 drugs and 958 cell lines. Keeping calibration strictly separate from training is what makes the conformal guarantee hold; reusing training data for calibration would break exchangeability and invalidate the coverage claim.

## Repository structure

```
.
├── notebooks/
│   └── drug_response_conformal.ipynb   # main notebook (add your Deepnote export here)
├── figures/                            # generated plots (EDA, calibration, intervals, residuals)
├── results/                            # metrics, predictions, coverage tables, LaTeX
│   ├── test_metrics.csv
│   ├── coverage_analysis.csv
│   ├── predictions.csv                 # per-sample prediction, interval, coverage flag
│   ├── experiment_summary.json
│   └── results_table.tex
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Large binaries, namely the cached GDSC pickle (`data/gdsc1.pkl`) and the model checkpoints (`*.pt`), are intentionally left untracked (see `.gitignore`). GDSC1 is re-downloaded automatically by TDC, and checkpoints can be regenerated by rerunning the notebook or attached to a GitHub Release if you want them available for download.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If the default wheels do not resolve, install `torch` and `torch-geometric` first with the build that matches your CUDA or CPU setup, following the official [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) instructions, then install the rest of `requirements.txt`.

## Usage

Open the notebook and run it top to bottom:

```bash
jupyter lab notebooks/drug_response_conformal.ipynb
```

GDSC1 is downloaded automatically through Therapeutics Data Commons on first run. The notebook then handles exploratory analysis, molecular featurization, model training with early stopping, conformal calibration across five alpha levels, the MC-dropout comparison, statistical tests on coverage, and export of all figures and metrics to `figures/` and `results/`.

## Dataset

GDSC1 (Genomics of Drug Sensitivity in Cancer) is accessed through [Therapeutics Data Commons](https://tdcommons.ai/). It contains 177,310 drug and cell-line sensitivity measurements spanning 208 drugs and 958 cell lines, with log(IC50) as the response variable. Around 89% of all possible drug and cell-line combinations are covered. Cell lines are represented by 17,737-gene expression profiles supplied alongside the response data.

## Limitations and future work

The current model treats each (drug, cell line) pair independently and uses a random split, so it measures interpolation performance rather than generalization to unseen drugs or unseen cell lines. Leave-drug-out and leave-cell-line-out splits would give a more demanding and clinically relevant test. On the uncertainty side, the conditional-coverage gap motivates moving from vanilla split conformal to normalized or locally adaptive conformal methods, or to conformalized quantile regression, which target more uniform coverage across the input space. Richer drug representations (attention-based or pretrained molecular encoders) and biologically informed gene selection are natural extensions on the modeling side.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Author

Yogarajan. Research focused on interpretable and uncertainty-aware machine learning for clinical and biomedical applications.
