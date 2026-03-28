import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_NODES, N_CHANNELS, D_MODEL, N_HEADS_GAT, N_TRANSFORMER_LAYERS, DROPOUT

class GeopoliticalEdgeGating(nn.Module):
    def __init__(self, ablation="A5"):
        super().__init__()
        self.ablation = ablation
        self.mlp_alpha = nn.Sequential(
            nn.Linear(N_CHANNELS, 16),
            nn.ReLU(),
            nn.Linear(16, N_CHANNELS)
        )
        self.W_g = nn.Parameter(torch.randn(N_NODES, N_NODES))
        self.b_g = nn.Parameter(torch.zeros(N_NODES, N_NODES))
        self.lambda_raw = nn.Parameter(torch.zeros(N_NODES, N_NODES))
        self.A_learned = nn.Parameter(torch.ones(N_NODES, N_NODES) * -2.0)  # Learned info graph

    def forward(self, gdelt_seq: torch.Tensor, a_static: torch.Tensor):
        B, k, N, _, C = gdelt_seq.shape
        lam = torch.sigmoid(self.lambda_raw)  # [5, 5]
        gate_prev = None
        
        a_dyn_seq = []
        gate_seq = []
        
        for t in range(k):
            e_t = gdelt_seq[:, t]  # [B, 5, 5, 3]
            
            if self.ablation == "A1":
                alpha = torch.ones_like(e_t) / N_CHANNELS
            else:
                alpha_logits = self.mlp_alpha(e_t)
                alpha = torch.softmax(alpha_logits, dim=-1)
            
            e_weighted = (alpha * e_t).sum(dim=-1)     # [B, 5, 5]
            gate_raw = torch.sigmoid(self.W_g.unsqueeze(0) * e_weighted + self.b_g.unsqueeze(0))
            
            lam_active = torch.ones_like(lam) if self.ablation == "A2" else lam
            
            if t == 0:
                gate_t = gate_raw
            else:
                gate_t = lam_active.unsqueeze(0) * gate_raw + (1.0 - lam_active.unsqueeze(0)) * gate_prev
            
            gate_prev = gate_t
            # Combine static physical trades with learned informational graph
            a_base = torch.clamp(a_static + torch.sigmoid(self.A_learned), 0, 1)
            a_dyn_t = a_base * gate_t  
            
            gate_seq.append(gate_t)
            a_dyn_seq.append(a_dyn_t)
            
        return torch.stack(a_dyn_seq, dim=1), torch.stack(gate_seq, dim=1)


class AsymmetricGraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int, ablation: str = "A5"):
        super().__init__()
        self.ablation = ablation
        assert out_features % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = out_features // n_heads
        self.W         = nn.Linear(in_features, out_features, bias=False)
        self.a         = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky     = nn.LeakyReLU(0.2)
        self.out_proj  = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        
        if self.ablation == "A3":
            adj = (adj + adj.transpose(-2, -1)) / 2.0
            
        Wx = self.W(x)                               
        Wx = Wx.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  

        a_l = self.a[:, :self.head_dim].unsqueeze(0).unsqueeze(2)    
        a_r = self.a[:, self.head_dim:].unsqueeze(0).unsqueeze(2)    
        e_i = (Wx * a_l).sum(-1, keepdim=True)                       
        e_j = (Wx * a_r).sum(-1, keepdim=True)                       
        
        e_raw = self.leaky(e_i + e_j.transpose(-2, -1))
        e = e_raw * adj.unsqueeze(1)
        
        mask = (adj.unsqueeze(1) == 0)
        e = e.masked_fill(mask, float('-inf'))
        attn = F.softmax(e, dim=-1)
        attn = attn.nan_to_num(0.0)
        
        h = (attn @ Wx).transpose(1, 2).contiguous().view(B, N, -1)    
        return F.elu(self.out_proj(h))


class GeoPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, timestep_offsets: torch.Tensor) -> torch.Tensor:
        k = x.size(1)
        return x + self.pe[:k, :].unsqueeze(0) + timestep_offsets


class TransformerTemporalEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, ablation: str = "A5"):
        super().__init__()
        self.ablation = ablation
        self.mlp_pe = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )
        self.geo_pe = GeoPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, sequence: torch.Tensor, gate_seq: torch.Tensor) -> torch.Tensor:
        B, k, _, _ = gate_seq.shape
        shock_intensity = gate_seq.view(B, k, -1).max(dim=-1)[0]  
        offsets = self.mlp_pe(shock_intensity.unsqueeze(-1))      
        
        if self.ablation == "A4":
            offsets = torch.zeros_like(offsets)
            
        x = self.geo_pe(sequence, offsets)
        z = self.encoder(x)
        return z[:, -1, :]  


class GeoRipNet(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads_gat: int = N_HEADS_GAT,
        n_transformer_layers: int = N_TRANSFORMER_LAYERS,
        dropout: float = DROPOUT,
        lookback: int = 20,
        ablation: str = "A5"
    ):
        super().__init__()
        self.lookback = lookback
        self.ablation = ablation

        self.edge_gating = GeopoliticalEdgeGating(ablation=ablation)

        self.price_embed = nn.Linear(1, d_model)
        self.gat = AsymmetricGraphAttentionLayer(
            in_features=d_model,
            out_features=d_model,
            n_heads=n_heads_gat,
            ablation=ablation
        )

        seq_dim = N_NODES * d_model
        self.seq_proj = nn.Linear(seq_dim, d_model)

        # Direct gate injection into Transformer sequence
        self.gate_proj = nn.Linear(N_NODES * N_NODES, d_model)

        self.temporal = TransformerTemporalEncoder(
            d_model=d_model,
            n_layers=n_transformer_layers,
            n_heads=min(4, d_model // 16),
            dropout=dropout,
            ablation=ablation
        )

        # Dual prediction heads
        # Head 1: log return regression (target = log(P_{t+1}/P_t))
        self.price_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, N_NODES),
        )
        # Head 2: direction classification (logit for P_{t+1} > P_t)
        self.dir_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, N_NODES),
        )

    def forward(
        self,
        prices: torch.Tensor,    # [B, k, N]
        gdelt: torch.Tensor,     # [B, k, N, N, C]
        a_static: torch.Tensor,  # [B, N, N]
    ):
        B, k, _ = prices.shape

        a_dyn_seq, gate_seq = self.edge_gating(gdelt, a_static)  # [B, k, N, N]

        spatial_seq = []
        for t in range(k):
            p_t = prices[:, t, :].unsqueeze(-1)           # [B, N, 1]
            h_t = self.price_embed(p_t)                   # [B, N, d]
            h_t = self.gat(h_t, a_dyn_seq[:, t, :, :])   # [B, N, d]
            spatial_seq.append(h_t.view(B, -1))            # [B, N*d]

        seq = torch.stack(spatial_seq, dim=1)              # [B, k, N*d]
        seq = self.seq_proj(seq)                           # [B, k, d]

        # Inject gate directly into Transformer input so geopolitics cannot be ignored
        gate_flat = gate_seq.view(B, k, N_NODES * N_NODES)  # [B, k, 25]
        seq = seq + self.gate_proj(gate_flat)               # [B, k, d]

        z = self.temporal(seq, gate_seq)   # [B, d]

        log_return = self.price_head(z)    # [B, N] — predicted log(P_{t+1}/P_t)
        dir_logit  = self.dir_head(z)      # [B, N] — raw logit for up/down direction
        return log_return, dir_logit
