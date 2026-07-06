"""Split conformal prediction for regression.

Extracted unchanged from the project notebook so the calibration, prediction,
and evaluation logic can be imported by the notebook, scripts, or tests.
"""

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm


class ConformalPredictor:
    """
    Split Conformal Prediction for regression.

    Provides prediction intervals with guaranteed coverage:
    P(Y in [y_hat - q, y_hat + q]) >= 1 - alpha

    where q is calibrated on a held-out calibration set.
    """

    def __init__(self, model, alpha=0.1):
        """
        Args:
            model: Trained prediction model
            alpha: Desired miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.model = model
        self.alpha = alpha
        self.quantile = None
        self.calibration_scores = None

    def calibrate(self, cal_loader, device):
        """
        Calibrate the conformal predictor on the calibration set.

        Computes nonconformity scores and finds the appropriate quantile.
        """
        self.model.eval()
        scores = []

        print(f"Calibrating with alpha = {self.alpha} (target coverage: {1-self.alpha:.0%})...")

        with torch.no_grad():
            for batch in tqdm(cal_loader, desc="Calibration"):
                batch = batch.to(device)
                predictions = self.model(batch)
                targets = batch.y.squeeze()

                # Nonconformity score = absolute residual
                residuals = torch.abs(targets - predictions)
                scores.extend(residuals.cpu().numpy())

        self.calibration_scores = np.array(scores)
        n = len(self.calibration_scores)

        # Compute quantile with finite-sample correction
        # This ensures valid coverage guarantee
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0

        self.quantile = np.quantile(self.calibration_scores, q_level)

        print("Calibration complete!")
        print(f"  - Calibration samples: {n}")
        print(f"  - Quantile level: {q_level:.4f}")
        print(f"  - Interval half-width (q): {self.quantile:.4f}")

        return self.quantile

    def predict(self, test_loader, device):
        """
        Generate predictions with conformal intervals.

        Returns dict with predictions, intervals, and targets.
        """
        self.model.eval()

        results = {
            'predictions': [],
            'lower': [],
            'upper': [],
            'targets': [],
            'interval_width': []
        }

        print("Generating predictions with conformal intervals...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Prediction"):
                batch = batch.to(device)
                predictions = self.model(batch)
                targets = batch.y.squeeze()

                for pred, target in zip(predictions, targets):
                    pred_val = pred.item()
                    target_val = target.item()

                    lower = pred_val - self.quantile
                    upper = pred_val + self.quantile

                    results['predictions'].append(pred_val)
                    results['lower'].append(lower)
                    results['upper'].append(upper)
                    results['targets'].append(target_val)
                    results['interval_width'].append(upper - lower)

        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])

        return results

    def evaluate(self, results):
        """
        Evaluate conformal predictor performance.
        """
        targets = results['targets']
        preds = results['predictions']
        lower = results['lower']
        upper = results['upper']
        widths = results['interval_width']

        # Coverage: fraction of true values within intervals
        covered = (targets >= lower) & (targets <= upper)
        coverage = covered.mean()

        # Prediction quality metrics
        rmse = np.sqrt(np.mean((targets - preds) ** 2))
        mae = np.mean(np.abs(targets - preds))
        pearson = stats.pearsonr(targets, preds)[0]
        spearman = stats.spearmanr(targets, preds)[0]

        # Interval quality metrics
        mean_width = widths.mean()
        median_width = np.median(widths)

        metrics = {
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'coverage_gap': abs(coverage - (1 - self.alpha)),
            'rmse': rmse,
            'mae': mae,
            'pearson': pearson,
            'spearman': spearman,
            'mean_interval_width': mean_width,
            'median_interval_width': median_width,
            'n_samples': len(targets)
        }

        return metrics
