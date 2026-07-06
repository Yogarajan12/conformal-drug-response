"""Minimal smoke tests for the drug_response package.

These do not check scientific accuracy (that comes from running the notebook).
They confirm the extracted pieces still wire together: featurization returns
the expected shape, the model runs a forward pass, and the conformal evaluator
computes coverage correctly. Run with `pytest` from the repository root.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from drug_response import (
    smiles_to_graph,
    DrugResponsePredictor,
    ConformalPredictor,
)

ATOM_FEATURE_DIM = 69


def test_featurization_shape():
    """A valid SMILES yields a graph with 69-dim atom features and some edges."""
    graph = smiles_to_graph("CCO")  # ethanol
    assert graph is not None
    assert graph.x.shape[1] == ATOM_FEATURE_DIM
    assert graph.x.shape[0] == 3           # C, C, O
    assert graph.edge_index.shape[0] == 2  # [2, num_edges]


def test_featurization_invalid_smiles_returns_none():
    assert smiles_to_graph("") is None
    assert smiles_to_graph("not_a_molecule") is None


def test_model_forward_shape():
    """The full predictor runs a forward pass and returns one value per graph."""
    n_genes = 32
    model = DrugResponsePredictor(
        drug_input_dim=ATOM_FEATURE_DIM, n_genes=n_genes,
        hidden_dim=16, n_gnn_layers=2, dropout=0.0,
    ).eval()

    graphs = []
    for _ in range(4):
        g = smiles_to_graph("CCO")
        g.cell_features = torch.randn(1, n_genes)
        g.y = torch.tensor([0.0])
        graphs.append(g)
    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4,)


def test_conformal_evaluate_coverage():
    """evaluate() computes coverage as the fraction of targets inside [lower, upper]."""
    cp = ConformalPredictor(model=None, alpha=0.1)
    preds = np.array([0.0, 0.0, 0.0, 0.0])
    q = 1.0
    results = {
        "predictions": preds,
        "lower": preds - q,
        "upper": preds + q,
        # 3 of 4 targets fall inside +/- 1.0
        "targets": np.array([0.5, -0.5, 0.9, 5.0]),
        "interval_width": np.full(4, 2 * q),
    }
    metrics = cp.evaluate(results)
    assert np.isclose(metrics["coverage"], 0.75)
    assert np.isclose(metrics["mean_interval_width"], 2.0)
    assert metrics["target_coverage"] == 0.9
