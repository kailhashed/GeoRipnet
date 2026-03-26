"""
model.py
GeoRipNet — Geopolitical Graph Neural Network for crude oil price forecasting.

Modules:
  1. GeopoliticalEdgeGating  — GDELT tensor → dynamic adjacency
  2. GraphAttentionLayer     — GAT spatial encoding per day
  3. TransformerTemporalEncoder — temporal context over lookback window
  4. GeoRipNet               — full model combining all modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_NODES, N_CHANNELS, D_MODEL, N_HEADS_GAT, N_TRANSFORMER_LAYERS, DROPOUT


# ── Module 1: Geopolitical Edge Gating ───────────────────────────────────────

class GeopoliticalEdgeGating(nn.Module):
    """
    Converts a GDELT tensor E(t) ∈ R^{5×5×3} into a dynamic adjacency mask.

    Gate(t) = σ(W_g · E(t) + b)     shape [5×5], values ∈ (0,1)
    A_dynamic(t) = A_static ⊙ Gate(t)
    """
    def __init__(self):
        super().__init__()
        self.W_g = nn.Linear(N_CHANNELS, 1, bias=True)

    def forward(self, gdelt: torch.Tensor, a_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gdelt:    [batch, 5, 5, 3]   GDELT tensor for day t
            a_static: [batch, 5, 5]      Comtrade adjacency for month of t
        Returns:
            a_dynamic: [batch, 5, 5]
        """
        gate = torch.sigmoid(self.W_g(gdelt).squeeze(-1))   # [batch, 5, 5]
        return a_static * gate


# ── Module 2: Graph Attention Layer ──────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """
    Single GAT layer with multi-head attention.
    Attention is masked to non-zero edges in a_dynamic.
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int):
        super().__init__()
        assert out_features % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = out_features // n_heads
        self.W         = nn.Linear(in_features, out_features, bias=False)
        self.a         = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky     = nn.LeakyReLU(0.2)
        self.out_proj  = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [batch, N, in_features]
            adj: [batch, N, N]  — dynamic adjacency (used as attention mask)
        Returns:
            h:   [batch, N, out_features]
        """
        B, N, _ = x.shape
        Wx = self.W(x)                               # [B, N, out]
        Wx = Wx.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]

        # Attention coefficients
        e_i = (Wx * self.a[:, :self.head_dim]).sum(-1, keepdim=True)   # [B,H,N,1]
        e_j = (Wx * self.a[:, self.head_dim:]).sum(-1, keepdim=True)   # [B,H,N,1]
        e   = self.leaky(e_i + e_j.transpose(-2, -1))                  # [B,H,N,N]

        # Mask: only attend along non-zero edges
        mask = (adj.unsqueeze(1) == 0)                                  # [B,1,N,N]
        e    = e.masked_fill(mask, float('-inf'))
        attn = F.softmax(e, dim=-1)
        attn = attn.nan_to_num(0.0)                                     # handle isolated nodes

        # Aggregate
        h = (attn @ Wx).transpose(1, 2).contiguous().view(B, N, -1)    # [B,N,out]
        return F.elu(self.out_proj(h))


# ── Module 3: Transformer Temporal Encoder ───────────────────────────────────

class TransformerTemporalEncoder(nn.Module):
    """
    Encodes a sequence of spatial snapshots [H(t-k), ..., H(t)] into
    a single context vector Z.
    """
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, 200, d_model))  # max seq 200
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [batch, k, d_model]   — flattened spatial snapshots
        Returns:
            Z: [batch, d_model]             — last timestep output
        """
        k = sequence.size(1)
        x = sequence + self.pos_enc[:, :k, :]
        z = self.encoder(x)
        return z[:, -1, :]   # take the last timestep as context


# ── Full Model ────────────────────────────────────────────────────────────────

class GeoRipNet(nn.Module):
    """
    Full GeoRipNet architecture.

    Input:
        prices:   [batch, k, 5]         historical prices
        gdelt:    [batch, 5, 5, 3]      GDELT tensor for day t
        a_static: [batch, 5, 5]         Comtrade adjacency for month of t

    Output:
        y_hat:    [batch, 5]            predicted next-day prices
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads_gat: int = N_HEADS_GAT,
        n_transformer_layers: int = N_TRANSFORMER_LAYERS,
        dropout: float = DROPOUT,
        lookback: int = 20,
    ):
        super().__init__()
        self.lookback = lookback

        # Module 1
        self.edge_gating = GeopoliticalEdgeGating()

        # Module 2 — price embedding + GAT
        self.price_embed = nn.Linear(1, d_model)
        self.gat = GraphAttentionLayer(
            in_features=d_model,
            out_features=d_model,
            n_heads=n_heads_gat
        )

        # Module 3
        seq_dim = N_NODES * d_model
        self.seq_proj = nn.Linear(seq_dim, d_model)   # project to d_model for transformer
        self.temporal = TransformerTemporalEncoder(
            d_model=d_model,
            n_layers=n_transformer_layers,
            n_heads=min(4, d_model // 16),
            dropout=dropout,
        )

        # Module 4
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, N_NODES),
        )

    def forward(
        self,
        prices: torch.Tensor,    # [B, k, 5]
        gdelt: torch.Tensor,     # [B, 5, 5, 3]
        a_static: torch.Tensor,  # [B, 5, 5]
    ) -> torch.Tensor:
        B, k, _ = prices.shape

        # Module 1: dynamic adjacency
        a_dyn = self.edge_gating(gdelt, a_static)  # [B, 5, 5]

        # Module 2: GAT over each day in the lookback window
        spatial_seq = []
        for t in range(k):
            p_t = prices[:, t, :].unsqueeze(-1)           # [B, 5, 1]
            h_t = self.price_embed(p_t)                    # [B, 5, d]
            h_t = self.gat(h_t, a_dyn)                    # [B, 5, d]
            spatial_seq.append(h_t.view(B, -1))           # [B, 5*d]

        seq = torch.stack(spatial_seq, dim=1)              # [B, k, 5*d]
        seq = self.seq_proj(seq)                           # [B, k, d]

        # Module 3: temporal encoding
        z = self.temporal(seq)                             # [B, d]

        # Module 4: predict
        return self.head(z)                                # [B, 5]
