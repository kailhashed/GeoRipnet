"""
RippleGNNLayer: Graph Neural Network for propagating shocks between countries.

This module implements Graph Attention Networks (GAT) to model how oil price
shocks in one country propagate to others through trade, supply, and sentiment links.

Mathematical Formulation:
    Δ(t+1) = α ⊙ Δ(t) + β (W ⊙ M) φ(Δ(t), E(t))
    
    where:
    - α: persistence parameters
    - β: propagation weight
    - W: trade adjacency matrix
    - M: learned attention mask
    - φ: message function
    - E: event/news embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention layer following Veličković et al. (2018).
    
    Computes attention coefficients between nodes and aggregates neighbor features.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        Args:
            h: (batch, num_nodes, in_features) - Node features
            adj: (batch, num_nodes, num_nodes) - Adjacency matrix (can be weighted)
        
        Returns:
            h_prime: (batch, num_nodes, out_features) - Updated node features
            attention: (batch, num_nodes, num_nodes) - Attention coefficients
        """
        batch_size, num_nodes, _ = h.shape
        
        # Linear transformation: (batch, num_nodes, out_features)
        Wh = torch.matmul(h, self.W)
        
        # Compute attention coefficients
        # Prepare for broadcast: (batch, num_nodes, 1, out_features)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        # (batch, 1, num_nodes, out_features)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        
        # Concatenate: (batch, num_nodes, num_nodes, 2*out_features)
        concat_features = torch.cat([Wh_i, Wh_j], dim=-1)
        
        # Compute attention logits: (batch, num_nodes, num_nodes)
        e = self.leakyrelu(torch.matmul(concat_features, self.a).squeeze(-1))
        
        # Mask attention for non-connected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax to get attention coefficients
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregate neighbor features: (batch, num_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention layer.
    
    Runs multiple attention heads in parallel and concatenates or averages results.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat_heads: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        # Create attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        # Output projection (if concatenating heads)
        if concat_heads:
            self.output_proj = nn.Linear(num_heads * out_features, out_features)
        
    def forward(self, h, adj):
        """
        Args:
            h: (batch, num_nodes, in_features)
            adj: (batch, num_nodes, num_nodes)
        
        Returns:
            h_out: (batch, num_nodes, out_features)
            attention_weights: List of attention matrices from each head
        """
        # Run each attention head
        head_outputs = []
        attention_weights = []
        
        for attn_head in self.attention_heads:
            h_prime, attn = attn_head(h, adj)
            head_outputs.append(h_prime)
            attention_weights.append(attn)
        
        # Concatenate or average heads
        if self.concat_heads:
            h_out = torch.cat(head_outputs, dim=-1)  # (batch, num_nodes, num_heads * out_features)
            h_out = self.output_proj(h_out)  # (batch, num_nodes, out_features)
        else:
            h_out = torch.stack(head_outputs, dim=0).mean(dim=0)  # Average
        
        return h_out, attention_weights


class RippleMessageFunction(nn.Module):
    """
    Message function φ(Δ_i(t), E_i(t)) for ripple propagation.
    
    Combines delta values with event/news embeddings to compute propagation messages.
    """
    
    def __init__(
        self,
        delta_dim: int = 1,
        event_embed_dim: int = 384,  # FinBERT/sentence-transformer embedding size
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.message_network = nn.Sequential(
            nn.Linear(delta_dim + event_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Gating mechanism to modulate message strength
        self.gate_network = nn.Sequential(
            nn.Linear(delta_dim + event_embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, deltas, event_embeddings):
        """
        Args:
            deltas: (batch, num_nodes, delta_dim) - Current delta values
            event_embeddings: (batch, num_nodes, event_embed_dim) - News embeddings
        
        Returns:
            messages: (batch, num_nodes, output_dim) - Propagation messages
        """
        # Concatenate deltas and event embeddings
        combined = torch.cat([deltas, event_embeddings], dim=-1)
        
        # Compute messages
        messages = self.message_network(combined)
        
        # Apply gating
        gate = self.gate_network(combined)
        messages = messages * gate
        
        return messages


class RippleGNNLayer(nn.Module):
    """
    Complete Ripple Propagation layer using Graph Attention Networks.
    
    Implements the full ripple propagation formula:
        Δ(t+1) = α ⊙ Δ(t) + β (W ⊙ M) φ(Δ(t), E(t))
    
    Args:
        num_countries: Number of countries (nodes in graph)
        delta_dim: Dimension of delta values (default: 1)
        event_embed_dim: Dimension of event embeddings (default: 384)
        hidden_dim: Hidden layer dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        num_gnn_layers: Number of GNN layers to stack (default: 2)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        num_countries: int,
        delta_dim: int = 1,
        event_embed_dim: int = 384,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_countries = num_countries
        self.delta_dim = delta_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Learnable persistence parameters α (per country)
        self.alpha = nn.Parameter(torch.ones(num_countries, delta_dim) * 0.5)
        
        # Learnable propagation weight β
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Message function
        self.message_function = RippleMessageFunction(
            delta_dim, event_embed_dim, hidden_dim, hidden_dim, dropout
        )
        
        # Stacked GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer: projects messages to hidden_dim
        self.gnn_layers.append(
            MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        )
        
        # Additional layers
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(
                MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            )
        
        # Output projection: hidden_dim -> delta_dim
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, delta_dim)
        )
        
        # Adaptive trade weight refinement
        self.trade_weight_refiner = nn.Sequential(
            nn.Linear(num_countries, num_countries),
            nn.GELU(),
            nn.Linear(num_countries, num_countries),
            nn.Sigmoid()
        )
        
    def refine_adjacency(self, trade_adjacency, learned_attention):
        """
        Combine static trade adjacency with learned attention.
        
        Args:
            trade_adjacency: (batch, num_nodes, num_nodes) - Trade-based adjacency
            learned_attention: (batch, num_nodes, num_nodes) - Learned attention
        
        Returns:
            refined_adj: (batch, num_nodes, num_nodes) - Combined adjacency
        """
        # Ensure trade adjacency is normalized
        trade_norm = trade_adjacency / (trade_adjacency.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Combine via element-wise product (Hadamard)
        refined_adj = trade_norm * learned_attention
        
        return refined_adj
    
    def forward(self, deltas, event_embeddings, trade_adjacency, return_attention=False):
        """
        Forward pass through ripple propagation layer.
        
        Args:
            deltas: (batch, num_countries, delta_dim) - Current delta values Δ(t)
            event_embeddings: (batch, num_countries, event_embed_dim) - News embeddings
            trade_adjacency: (batch, num_countries, num_countries) - Trade weight matrix W
            return_attention: If True, return attention weights
        
        Returns:
            deltas_next: (batch, num_countries, delta_dim) - Propagated deltas Δ(t+1)
            attention_weights: Optional list of attention matrices
        """
        batch_size = deltas.size(0)
        
        # Compute messages φ(Δ(t), E(t))
        messages = self.message_function(deltas, event_embeddings)  # (batch, num_countries, hidden_dim)
        
        # Propagate through GNN layers
        h = messages
        all_attention_weights = []
        
        for gnn_layer in self.gnn_layers:
            h, attn_weights = gnn_layer(h, trade_adjacency)
            all_attention_weights.extend(attn_weights)
        
        # Project back to delta dimension
        propagated = self.output_projection(h)  # (batch, num_countries, delta_dim)
        
        # Apply ripple propagation formula: Δ(t+1) = α ⊙ Δ(t) + β * propagated
        alpha = torch.sigmoid(self.alpha).unsqueeze(0)  # (1, num_countries, delta_dim)
        deltas_next = alpha * deltas + self.beta * propagated
        
        if return_attention:
            return deltas_next, all_attention_weights
        else:
            return deltas_next
    
    def compute_ripple_influence(self, deltas, trade_adjacency):
        """
        Compute ripple influence matrix showing how much each country affects others.
        
        Args:
            deltas: (batch, num_countries, delta_dim)
            trade_adjacency: (batch, num_countries, num_countries)
        
        Returns:
            influence_matrix: (batch, num_countries, num_countries)
        """
        batch_size, num_countries, _ = deltas.shape
        
        # Compute pairwise influence scores
        # Broadcast deltas: (batch, num_countries, 1, delta_dim)
        deltas_i = deltas.unsqueeze(2).expand(-1, -1, num_countries, -1)
        # (batch, 1, num_countries, delta_dim)
        deltas_j = deltas.unsqueeze(1).expand(-1, num_countries, -1, -1)
        
        # Compute absolute difference (shock magnitude)
        shock_diff = torch.abs(deltas_i - deltas_j).squeeze(-1)  # (batch, num_countries, num_countries)
        
        # Weight by trade adjacency
        influence = shock_diff * trade_adjacency
        
        return influence


class TemporalRippleGNN(nn.Module):
    """
    Temporal extension of RippleGNN that processes sequences of deltas.
    
    Useful for modeling multi-step ripple propagation dynamics.
    """
    
    def __init__(
        self,
        num_countries: int,
        seq_len: int = 7,  # Number of time steps to propagate
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_countries = num_countries
        
        # Base ripple layer
        self.ripple_layer = RippleGNNLayer(num_countries, **kwargs)
        
        # Temporal aggregation (LSTM)
        hidden_dim = kwargs.get('hidden_dim', 128)
        self.temporal_lstm = nn.LSTM(
            input_size=1,  # delta_dim
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=kwargs.get('dropout', 0.3),
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
    
    def forward(self, delta_sequence, event_sequence, trade_adjacency):
        """
        Args:
            delta_sequence: (batch, seq_len, num_countries, delta_dim)
            event_sequence: (batch, seq_len, num_countries, event_embed_dim)
            trade_adjacency: (batch, num_countries, num_countries)
        
        Returns:
            final_deltas: (batch, num_countries, delta_dim)
        """
        batch_size, seq_len, num_countries, delta_dim = delta_sequence.shape
        
        # Process each time step
        propagated_sequence = []
        
        for t in range(seq_len):
            deltas_t = delta_sequence[:, t, :, :]  # (batch, num_countries, delta_dim)
            events_t = event_sequence[:, t, :, :]  # (batch, num_countries, event_embed_dim)
            
            # Propagate
            deltas_next = self.ripple_layer(deltas_t, events_t, trade_adjacency)
            propagated_sequence.append(deltas_next)
        
        # Stack: (batch, seq_len, num_countries, delta_dim)
        propagated_sequence = torch.stack(propagated_sequence, dim=1)
        
        # Temporal aggregation via LSTM
        # Reshape: (batch * num_countries, seq_len, delta_dim)
        lstm_input = propagated_sequence.view(batch_size * num_countries, seq_len, delta_dim)
        lstm_out, _ = self.temporal_lstm(lstm_input)  # (batch * num_countries, seq_len, hidden_dim)
        
        # Take last time step
        final_hidden = lstm_out[:, -1, :]  # (batch * num_countries, hidden_dim)
        
        # Project to delta
        final_deltas = self.output_projection(final_hidden)  # (batch * num_countries, delta_dim)
        
        # Reshape back: (batch, num_countries, delta_dim)
        final_deltas = final_deltas.view(batch_size, num_countries, delta_dim)
        
        return final_deltas


if __name__ == "__main__":
    # Test RippleGNNLayer
    print("Testing RippleGNNLayer...")
    
    batch_size = 8
    num_countries = 20
    delta_dim = 1
    event_embed_dim = 384
    
    model = RippleGNNLayer(
        num_countries=num_countries,
        delta_dim=delta_dim,
        event_embed_dim=event_embed_dim,
        hidden_dim=128,
        num_heads=4,
        num_gnn_layers=2,
        dropout=0.3
    )
    
    # Random inputs
    deltas = torch.randn(batch_size, num_countries, delta_dim)
    event_embeddings = torch.randn(batch_size, num_countries, event_embed_dim)
    trade_adjacency = torch.rand(batch_size, num_countries, num_countries)
    
    # Make adjacency symmetric and add self-loops
    trade_adjacency = (trade_adjacency + trade_adjacency.transpose(1, 2)) / 2
    eye = torch.eye(num_countries).unsqueeze(0).expand(batch_size, -1, -1)
    trade_adjacency = trade_adjacency + eye
    
    # Forward pass
    deltas_next, attention_weights = model(
        deltas, event_embeddings, trade_adjacency, return_attention=True
    )
    
    print(f"Input deltas shape: {deltas.shape}")
    print(f"Output deltas shape: {deltas_next.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    
    # Test influence computation
    influence = model.compute_ripple_influence(deltas, trade_adjacency)
    print(f"Influence matrix shape: {influence.shape}")
    
    # Test TemporalRippleGNN
    print("\nTesting TemporalRippleGNN...")
    temporal_model = TemporalRippleGNN(
        num_countries=num_countries,
        seq_len=7,
        delta_dim=delta_dim,
        event_embed_dim=event_embed_dim,
        hidden_dim=128,
        dropout=0.3
    )
    
    seq_len = 7
    delta_sequence = torch.randn(batch_size, seq_len, num_countries, delta_dim)
    event_sequence = torch.randn(batch_size, seq_len, num_countries, event_embed_dim)
    
    final_deltas = temporal_model(delta_sequence, event_sequence, trade_adjacency)
    print(f"Temporal output deltas shape: {final_deltas.shape}")
    
    print("\n✓ RippleGNNLayer test passed!")

