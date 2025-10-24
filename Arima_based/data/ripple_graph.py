"""
Ripple Graph Module for RippleNet-TFT
Builds country-commodity graph and propagates geopolitical shocks
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch Geometric imports (with fallback)
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data, DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Fallback implementations
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index, edge_attr=None):
            return self.linear(x)
    
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index, edge_attr=None):
            return self.linear(x)
    
    class SAGEConv(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
        def forward(self, x, edge_index, edge_attr=None):
            return self.linear(x)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RippleGraph:
    """Ripple Graph for propagating geopolitical shocks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.DiGraph()
        self.country_commodity_map = {}
        self.trade_weights = {}
        self.embedding_dim = config['model']['ripple_graph']['embedding_dim']
        
    def build_country_commodity_graph(self, trade_data: Optional[pd.DataFrame] = None) -> nx.DiGraph:
        """Build country-commodity graph from trade data"""
        logger.info("Building country-commodity graph")
        
        # If no trade data provided, create synthetic trade relationships
        if trade_data is None or trade_data.empty:
            logger.warning("No trade data provided, creating synthetic relationships")
            trade_data = self._create_synthetic_trade_data()
        
        # Add nodes
        countries = trade_data['country'].unique()
        commodities = trade_data['commodity'].unique()
        
        for country in countries:
            self.graph.add_node(country, node_type='country')
        
        for commodity in commodities:
            self.graph.add_node(commodity, node_type='commodity')
        
        # Add edges with trade weights
        for _, row in trade_data.iterrows():
            country = row['country']
            commodity = row['commodity']
            weight = row['trade_volume']
            
            self.graph.add_edge(country, commodity, weight=weight)
            self.graph.add_edge(commodity, country, weight=weight * 0.1)  # Reverse flow
        
        # Normalize edge weights
        self._normalize_edge_weights()
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _create_synthetic_trade_data(self) -> pd.DataFrame:
        """Create synthetic trade data for demonstration"""
        countries = ['USA', 'China', 'Russia', 'Saudi_Arabia', 'Iran', 'Venezuela', 'Canada', 'Norway']
        commodities = ['crude_oil', 'natural_gas', 'coal', 'electricity']
        
        trade_data = []
        np.random.seed(42)
        
        for country in countries:
            for commodity in commodities:
                # Create realistic trade relationships
                if country == 'Saudi_Arabia' and commodity == 'crude_oil':
                    volume = np.random.uniform(1000, 2000)
                elif country == 'Russia' and commodity in ['crude_oil', 'natural_gas']:
                    volume = np.random.uniform(800, 1500)
                elif country == 'USA' and commodity in ['crude_oil', 'natural_gas', 'coal']:
                    volume = np.random.uniform(500, 1200)
                elif country == 'China' and commodity in ['crude_oil', 'coal']:
                    volume = np.random.uniform(600, 1400)
                else:
                    volume = np.random.uniform(100, 800)
                
                trade_data.append({
                    'country': country,
                    'commodity': commodity,
                    'trade_volume': volume
                })
        
        return pd.DataFrame(trade_data)
    
    def _normalize_edge_weights(self):
        """Normalize edge weights to [0, 1] range"""
        edges = list(self.graph.edges(data=True))
        weights = [edge[2]['weight'] for edge in edges]
        
        if weights:
            min_weight = min(weights)
            max_weight = max(weights)
            
            for edge in edges:
                normalized_weight = (edge[2]['weight'] - min_weight) / (max_weight - min_weight)
                self.graph[edge[0]][edge[1]]['weight'] = normalized_weight
    
    def compute_daily_impact(self, gdelt_data: pd.DataFrame) -> pd.DataFrame:
        """Compute daily geopolitical impact per country"""
        logger.info("Computing daily geopolitical impact")
        
        if gdelt_data.empty:
            logger.warning("No GDELT data provided, creating synthetic impact data")
            return self._create_synthetic_impact_data()
        
        # Group by date and country
        impact_df = gdelt_data.groupby(['date', 'country']).agg({
            'tone': 'mean',
            'goldstein_scale': 'mean',
            'title': 'count'
        }).reset_index()
        
        # Calculate impact score
        impact_df['impact_score'] = (
            impact_df['tone'] * impact_df['goldstein_scale'] * impact_df['title']
        )
        
        # Normalize impact scores
        impact_df['impact_score'] = (impact_df['impact_score'] - impact_df['impact_score'].min()) / (
            impact_df['impact_score'].max() - impact_df['impact_score'].min()
        )
        
        logger.info(f"Computed impact for {len(impact_df)} country-date combinations")
        return impact_df
    
    def _create_synthetic_impact_data(self) -> pd.DataFrame:
        """Create synthetic impact data for demonstration"""
        countries = ['USA', 'China', 'Russia', 'Saudi_Arabia', 'Iran', 'Venezuela', 'Canada', 'Norway']
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        
        impact_data = []
        np.random.seed(42)
        
        for date in dates:
            for country in countries:
                # Create realistic impact patterns
                base_impact = np.random.normal(0.5, 0.2)
                
                # Add some geopolitical events
                if country == 'Russia' and date.year >= 2022:
                    base_impact += np.random.uniform(0.3, 0.8)  # Ukraine conflict
                elif country == 'Saudi_Arabia' and date.month in [3, 4, 10, 11]:
                    base_impact += np.random.uniform(0.2, 0.6)  # OPEC meetings
                elif country == 'Iran' and date.year >= 2021:
                    base_impact += np.random.uniform(0.1, 0.4)  # Sanctions
                
                impact_data.append({
                    'date': date,
                    'country': country,
                    'impact_score': max(0, min(1, base_impact))
                })
        
        return pd.DataFrame(impact_data)
    
    def propagate_ripples(self, impact_data: pd.DataFrame, 
                         method: str = 'gnn') -> pd.DataFrame:
        """Propagate geopolitical ripples through the graph"""
        logger.info(f"Propagating ripples using {method} method")
        
        if method == 'gnn':
            return self._propagate_with_gnn(impact_data)
        elif method == 'diffusion':
            return self._propagate_with_diffusion(impact_data)
        else:
            raise ValueError(f"Unknown propagation method: {method}")
    
    def _propagate_with_gnn(self, impact_data: pd.DataFrame) -> pd.DataFrame:
        """Propagate ripples using Graph Neural Network"""
        logger.info("Propagating ripples with GNN")
        
        # Convert NetworkX graph to PyTorch Geometric format
        edge_index, edge_weights = self._nx_to_pyg()
        
        # Create GNN model
        gnn_model = RippleGNN(
            input_dim=1,
            hidden_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            num_layers=self.config['model']['ripple_graph']['num_layers']
        )
        
        # Process each date
        ripple_embeddings = []
        
        for date in impact_data['date'].unique():
            date_impacts = impact_data[impact_data['date'] == date]
            
            # Create node features (impact scores)
            node_features = torch.zeros(self.graph.number_of_nodes(), 1)
            node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
            
            for _, row in date_impacts.iterrows():
                if row['country'] in node_to_idx:
                    node_features[node_to_idx[row['country']]] = row['impact_score']
            
            # Create PyG data object
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = gnn_model(data.x, data.edge_index, data.edge_attr)
            
            # Store embeddings for commodities
            commodity_embeddings = {}
            for node, idx in node_to_idx.items():
                if self.graph.nodes[node]['node_type'] == 'commodity':
                    commodity_embeddings[node] = embeddings[idx].numpy()
            
            ripple_embeddings.append({
                'date': date,
                'embeddings': commodity_embeddings
            })
        
        return pd.DataFrame(ripple_embeddings)
    
    def _propagate_with_diffusion(self, impact_data: pd.DataFrame) -> pd.DataFrame:
        """Propagate ripples using simple diffusion"""
        logger.info("Propagating ripples with diffusion")
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph, weight='weight').toarray()
        
        # Normalize adjacency matrix
        degree = np.sum(adj_matrix, axis=1)
        degree[degree == 0] = 1  # Avoid division by zero
        adj_matrix = adj_matrix / degree[:, np.newaxis]
        
        ripple_embeddings = []
        
        for date in impact_data['date'].unique():
            date_impacts = impact_data[impact_data['date'] == date]
            
            # Create initial impact vector
            impact_vector = np.zeros(self.graph.number_of_nodes())
            node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
            
            for _, row in date_impacts.iterrows():
                if row['country'] in node_to_idx:
                    impact_vector[node_to_idx[row['country']]] = row['impact_score']
            
            # Propagate through diffusion
            for _ in range(3):  # Number of diffusion steps
                impact_vector = adj_matrix @ impact_vector
            
            # Extract commodity embeddings
            commodity_embeddings = {}
            for node, idx in node_to_idx.items():
                if self.graph.nodes[node]['node_type'] == 'commodity':
                    commodity_embeddings[node] = impact_vector[idx]
            
            ripple_embeddings.append({
                'date': date,
                'embeddings': commodity_embeddings
            })
        
        return pd.DataFrame(ripple_embeddings)
    
    def _nx_to_pyg(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert NetworkX graph to PyTorch Geometric format"""
        edge_list = list(self.graph.edges())
        edge_index = torch.tensor(edge_list).t().contiguous()
        
        edge_weights = torch.tensor([
            self.graph[edge[0]][edge[1]]['weight'] for edge in edge_list
        ]).float()
        
        return edge_index, edge_weights
    
    def create_ripple_dataset(self, impact_data: pd.DataFrame, 
                            method: str = 'gnn') -> pd.DataFrame:
        """Create ripple dataset for training"""
        logger.info("Creating ripple dataset")
        
        # Propagate ripples
        ripple_data = self.propagate_ripples(impact_data, method)
        
        # Flatten embeddings into columns
        flattened_data = []
        
        for _, row in ripple_data.iterrows():
            date = row['date']
            embeddings = row['embeddings']
            
            row_data = {'date': date}
            
            for commodity, embedding in embeddings.items():
                if isinstance(embedding, np.ndarray):
                    for i, val in enumerate(embedding):
                        row_data[f'{commodity}_ripple_{i}'] = val
                else:
                    row_data[f'{commodity}_ripple'] = embedding
            
            flattened_data.append(row_data)
        
        return pd.DataFrame(flattened_data)

class RippleGNN(nn.Module):
    """Graph Neural Network for ripple propagation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(RippleGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.layers.append(GCNConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if TORCH_GEOMETRIC_AVAILABLE:
                x = layer(x, edge_index)
            else:
                # Fallback: just use linear transformation
                x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

def main():
    """Main function for ripple graph processing"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ripple graph
    ripple_graph = RippleGraph(config)
    
    # Build graph
    graph = ripple_graph.build_country_commodity_graph()
    
    # Create synthetic impact data
    impact_data = ripple_graph._create_synthetic_impact_data()
    
    # Create ripple dataset
    ripple_dataset = ripple_graph.create_ripple_dataset(impact_data, method='diffusion')
    
    print(f"Ripple dataset created with shape: {ripple_dataset.shape}")
    print("Ripple graph processing completed successfully!")

if __name__ == "__main__":
    main()
