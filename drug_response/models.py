"""Model definitions: drug GNN encoder, cell-line MLP encoder, and the combined
drug-response predictor.

Extracted unchanged from the project notebook so the architecture can be
imported by the notebook, scripts, or tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class DrugEncoder(nn.Module):
    """
    Graph Neural Network encoder for drug molecules.
    Uses GCN layers with residual connections.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, dropout=0.2):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GCN layers with residual connections
        for i in range(self.n_layers):
            residual = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection

        # Global pooling (combine mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = x_mean + x_max

        # Output projection
        x = self.output_proj(x)

        return x


class CellLineEncoder(nn.Module):
    """
    MLP encoder for cell line gene expression features.
    Transforms high-dimensional gene expression to dense representation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()

        # Dimensionality reduction MLP
        # input_dim is number of genes, output_dim is embedding size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, gene_expression):
        """
        Args:
            gene_expression: (batch_size, n_genes) tensor
        Returns:
            (batch_size, output_dim) tensor
        """
        return self.encoder(gene_expression)


class DrugResponsePredictor(nn.Module):
    """
    Complete model for drug response prediction.
    Combines drug GNN encoder and cell line gene expression encoder.
    """

    def __init__(self, drug_input_dim, n_genes, hidden_dim=128,
                 n_gnn_layers=3, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Drug encoder (GNN)
        self.drug_encoder = DrugEncoder(
            input_dim=drug_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_gnn_layers,
            dropout=dropout
        )

        # Cell line encoder (MLP on gene expression)
        self.cell_encoder = CellLineEncoder(
            input_dim=n_genes,        # Number of genes
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        # Encode drug (using GNN)
        drug_emb = self.drug_encoder(data.x, data.edge_index, data.batch)

        # Encode cell line (using gene expression MLP)
        # data.cell_features has shape (batch_size, n_genes)
        cell_emb = self.cell_encoder(data.cell_features)

        # Combine and predict
        combined = torch.cat([drug_emb, cell_emb], dim=1)
        output = self.predictor(combined)

        return output.squeeze()
