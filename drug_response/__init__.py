"""Core reusable components for the uncertainty-aware drug response project.

The full experiment (data loading, training loop, plotting) lives in the
notebook. This package holds the pieces worth importing and reusing:
molecular featurization, the model architecture, and the split conformal
predictor.
"""

from .featurization import get_atom_features, smiles_to_graph
from .models import DrugEncoder, CellLineEncoder, DrugResponsePredictor
from .conformal import ConformalPredictor

__all__ = [
    "get_atom_features",
    "smiles_to_graph",
    "DrugEncoder",
    "CellLineEncoder",
    "DrugResponsePredictor",
    "ConformalPredictor",
]
