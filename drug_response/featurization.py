"""Molecular featurization: convert SMILES strings into PyTorch Geometric graphs.

Extracted unchanged from the project notebook so the same featurization can be
imported by the notebook, scripts, or tests.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data


def get_atom_features(atom):
    """
    Extract features for a single atom.
    Returns a feature vector of length 69.
      - Atom type one-hot: 44
      - Degree one-hot: 11
      - Formal charge one-hot: 5
      - Hybridization one-hot: 7
      - Is aromatic: 1
      - Normalized num Hs: 1
    """
    # Atom type (one-hot, 44 possible elements)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                  'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                  'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                  'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                  'Pt', 'Hg', 'Pb', 'Other']

    atom_symbol = atom.GetSymbol()
    atom_type_encoding = [0] * len(atom_types)
    if atom_symbol in atom_types:
        atom_type_encoding[atom_types.index(atom_symbol)] = 1
    else:
        atom_type_encoding[-1] = 1  # 'Other'

    # Degree (one-hot, 0-10)
    degree = min(atom.GetDegree(), 10)
    degree_encoding = [0] * 11
    degree_encoding[degree] = 1

    # Formal charge (one-hot, -2 to 2)
    formal_charge = max(-2, min(atom.GetFormalCharge(), 2)) + 2
    charge_encoding = [0] * 5
    charge_encoding[formal_charge] = 1

    # Hybridization (one-hot)
    hybridization_types = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hybridization = atom.GetHybridization()
    hybridization_encoding = [0] * (len(hybridization_types) + 1)
    if hybridization in hybridization_types:
        hybridization_encoding[hybridization_types.index(hybridization)] = 1
    else:
        hybridization_encoding[-1] = 1

    # Additional features
    is_aromatic = [1 if atom.GetIsAromatic() else 0]
    num_hs = [min(atom.GetTotalNumHs(), 4) / 4.0]  # Normalized

    # Combine all features
    features = (atom_type_encoding + degree_encoding + charge_encoding +
                hybridization_encoding + is_aromatic + num_hs)

    return features


def smiles_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric Data object.
    Returns None if conversion fails.
    """
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Check if molecule has atoms
        if mol.GetNumAtoms() == 0:
            return None

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))

        x = torch.tensor(atom_features, dtype=torch.float)

        # Get edge indices (bonds)
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        if len(edge_indices) == 0:
            # Single atom molecule - add self-loop
            edge_indices = [[0, 0]]

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    except Exception:
        # Silently return None for any parsing errors
        return None
