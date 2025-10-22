"""
Graph Attention Network (GAT) for Knowledge Graph Fusion
This module implements the third component of the proposed framework:
Fuse knowledge graphs using Attention-based Fusion (Graph Attention Network-GAT) to align relations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from knowledge_graph import HealthcareKnowledgeGraph, GraphNode, GraphEdge, NodeType

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformations for attention mechanism
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # LeakyReLU for attention
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention layer
        
        Args:
            h: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
            
        Returns:
            output: Updated node features [N, out_features]
            attention: Attention weights [N, N]
        """
        N = h.size(0)
        
        # Linear transformation
        Wh = self.W(h)  # [N, out_features]
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input))  # [N*N, 1]
        e = e.view(N, N)  # [N, N]
        
        # Apply attention mask (only attend to connected nodes)
        attention = torch.where(adj > 0, e, torch.tensor(-9e15, dtype=e.dtype, device=e.device))
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        
        return h_prime, attention
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for attention mechanism
        """
        N = Wh.size(0)
        
        # Create all possible combinations of node pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        return all_combinations_matrix

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head Graph Attention Network
    """
    
    def __init__(self, n_heads: int, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha)
            for _ in range(n_heads)
        ])
        
        # Final linear layer
        self.final_linear = nn.Linear(n_heads * out_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Multi-head attention forward pass
        """
        head_outputs = []
        attention_weights = []
        
        for attention in self.attentions:
            head_out, attn = attention(h, adj)
            head_outputs.append(head_out)
            attention_weights.append(attn)
        
        # Concatenate all heads
        multi_head_output = torch.cat(head_outputs, dim=1)
        multi_head_output = self.dropout(multi_head_output)
        
        # Final linear transformation
        output = self.final_linear(multi_head_output)
        
        return output, attention_weights

class GATFusion(nn.Module):
    """
    Graph Attention Network for Knowledge Graph Fusion
    """
    
    def __init__(self, 
                 node_features: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 alpha: float = 0.2):
        super(GATFusion, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(node_features, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim
            
            self.gat_layers.append(
                MultiHeadGraphAttention(n_heads, in_dim, hidden_dim, dropout, alpha)
            )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        Forward pass through the GAT fusion network
        
        Args:
            node_features: Node feature matrix [N, node_features]
            adjacency_matrix: Adjacency matrix [N, N]
            
        Returns:
            output: Fused node representations [N, output_dim]
            attention_weights: Attention weights for each layer and head
        """
        # Input projection
        h = self.input_projection(node_features)
        h = self.dropout(h)
        
        all_attention_weights = []
        
        # Pass through GAT layers
        for gat_layer in self.gat_layers:
            h_new, attention_weights = gat_layer(h, adjacency_matrix)
            h = self.layer_norm(h + h_new)  # Residual connection
            all_attention_weights.append(attention_weights)
        
        # Output projection
        output = self.output_projection(h)
        
        return output, all_attention_weights

class KnowledgeGraphFusion:
    """
    Main class for fusing knowledge graphs using GAT
    """
    
    def __init__(self, 
                 node_feature_dim: int = 128,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 n_heads: int = 4,
                 n_layers: int = 2):
        """
        Initialize the fusion system
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden dimension for GAT
            output_dim: Output dimension for fused representations
            n_heads: Number of attention heads
            n_layers: Number of GAT layers
        """
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Initialize GAT model
        self.gat_model = GATFusion(
            node_features=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # Node and edge encoders
        self.node_encoder = NodeEncoder(node_feature_dim)
        self.edge_encoder = EdgeEncoder()
    
    def prepare_graph_data(self, knowledge_graph: HealthcareKnowledgeGraph) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Prepare graph data for GAT processing
        
        Args:
            knowledge_graph: The knowledge graph to process
            
        Returns:
            node_features: Node feature matrix [N, node_feature_dim]
            adjacency_matrix: Adjacency matrix [N, N]
            node_mapping: Mapping from node IDs to indices
        """
        # Get all nodes
        nodes = list(knowledge_graph.graph.nodes())
        node_mapping = {node_id: idx for idx, node_id in enumerate(nodes)}
        
        # Create node features
        node_features = []
        for node_id in nodes:
            node_data = knowledge_graph.graph.nodes[node_id]
            features = self.node_encoder.encode_node(node_data)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Create adjacency matrix
        N = len(nodes)
        adjacency_matrix = torch.zeros(N, N)
        
        for edge in knowledge_graph.graph.edges(data=True):
            source_idx = node_mapping[edge[0]]
            target_idx = node_mapping[edge[1]]
            weight = edge[2].get('weight', 1.0)
            adjacency_matrix[source_idx, target_idx] = weight
        
        return node_features, adjacency_matrix, node_mapping
    
    def fuse_graphs(self, 
                   primary_graph: HealthcareKnowledgeGraph,
                   secondary_graph: HealthcareKnowledgeGraph = None) -> HealthcareKnowledgeGraph:
        """
        Fuse multiple knowledge graphs using GAT
        
        Args:
            primary_graph: Main knowledge graph
            secondary_graph: Secondary graph to fuse (optional)
            
        Returns:
            Fused knowledge graph
        """
        print("Starting graph fusion process...")
        
        # Prepare primary graph data
        node_features, adjacency_matrix, node_mapping = self.prepare_graph_data(primary_graph)
        
        # If secondary graph provided, merge them
        if secondary_graph:
            node_features, adjacency_matrix, node_mapping = self._merge_graphs(
                primary_graph, secondary_graph, node_features, adjacency_matrix, node_mapping
            )
        
        # Apply GAT fusion
        with torch.no_grad():
            fused_features, attention_weights = self.gat_model(node_features, adjacency_matrix)
        
        # Create fused graph
        fused_graph = self._create_fused_graph(
            primary_graph, fused_features, node_mapping, attention_weights
        )
        
        print(f"Graph fusion completed. Fused graph has {fused_graph.graph.number_of_nodes()} nodes and {fused_graph.graph.number_of_edges()} edges")
        
        return fused_graph
    
    def _merge_graphs(self, 
                     primary_graph: HealthcareKnowledgeGraph,
                     secondary_graph: HealthcareKnowledgeGraph,
                     node_features: torch.Tensor,
                     adjacency_matrix: torch.Tensor,
                     node_mapping: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Merge two knowledge graphs"""
        # Get secondary graph nodes
        secondary_nodes = list(secondary_graph.graph.nodes())
        
        # Update node mapping
        current_max_idx = len(node_mapping)
        for node_id in secondary_nodes:
            if node_id not in node_mapping:
                node_mapping[node_id] = current_max_idx
                current_max_idx += 1
        
        # Extend node features
        secondary_features = []
        for node_id in secondary_nodes:
            node_data = secondary_graph.graph.nodes[node_id]
            features = self.node_encoder.encode_node(node_data)
            secondary_features.append(features)
        
        secondary_features = torch.tensor(secondary_features, dtype=torch.float32)
        node_features = torch.cat([node_features, secondary_features], dim=0)
        
        # Extend adjacency matrix
        N = node_features.size(0)
        new_adjacency = torch.zeros(N, N)
        new_adjacency[:adjacency_matrix.size(0), :adjacency_matrix.size(1)] = adjacency_matrix
        
        # Add secondary graph edges
        for edge in secondary_graph.graph.edges(data=True):
            source_idx = node_mapping[edge[0]]
            target_idx = node_mapping[edge[1]]
            weight = edge[2].get('weight', 1.0)
            new_adjacency[source_idx, target_idx] = weight
        
        return node_features, new_adjacency, node_mapping
    
    def _create_fused_graph(self, 
                           original_graph: HealthcareKnowledgeGraph,
                           fused_features: torch.Tensor,
                           node_mapping: Dict,
                           attention_weights: List[List[torch.Tensor]]) -> HealthcareKnowledgeGraph:
        """Create the fused knowledge graph"""
        fused_graph = HealthcareKnowledgeGraph()
        
        # Add nodes with fused features
        for node_id, idx in node_mapping.items():
            if node_id in original_graph.graph.nodes():
                node_data = original_graph.graph.nodes[node_id].copy()
                node_data['fused_embedding'] = fused_features[idx].numpy()
                fused_graph.add_node(GraphNode(
                    id=node_id,
                    node_type=NodeType(node_data.get('node_type', 'disease')),
                    name=node_data.get('name', node_id),
                    attributes=node_data,
                    embedding=fused_features[idx].numpy()
                ))
        
        # Add edges from original graph
        for edge in original_graph.graph.edges(data=True):
            edge_obj = GraphEdge(
                source=edge[0],
                target=edge[1],
                relation_type=edge[2].get('relation_type', ''),
                weight=edge[2].get('weight', 1.0),
                confidence=edge[2].get('confidence', 1.0),
                metadata=edge[2].get('metadata', {})
            )
            fused_graph.add_edge(edge_obj)
        
        return fused_graph
    
    def save_model(self, filepath: str):
        """Save the trained GAT model"""
        torch.save({
            'model_state_dict': self.gat_model.state_dict(),
            'node_feature_dim': self.node_feature_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers
        }, filepath)
        print(f"GAT model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained GAT model"""
        checkpoint = torch.load(filepath)
        self.gat_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"GAT model loaded from {filepath}")

# ------------------------------
# Minimal helpers to load/save/train weights externally
# ------------------------------

import os

def load_gat_weights(path: str):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
    except Exception:
        pass
    return None

def save_gat_weights(state_dict, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        return True
    except Exception:
        return False

def train_gat_stub(knowledge_graph: HealthcareKnowledgeGraph) -> dict:
    """Placeholder that returns an empty state dict. Replace with real training."""
    return {}

class NodeEncoder:
    """Encode nodes into feature vectors"""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.node_type_embeddings = self._create_node_type_embeddings()
    
    def _create_node_type_embeddings(self) -> Dict[str, torch.Tensor]:
        """Create embeddings for different node types"""
        node_types = ['disease', 'symptom', 'drug', 'lifestyle', 'nutrition', 'prevention', 'severity']
        embeddings = {}
        
        for i, node_type in enumerate(node_types):
            # Create a one-hot encoding for node type
            embedding = torch.zeros(self.feature_dim)
            embedding[i * (self.feature_dim // len(node_types)):(i + 1) * (self.feature_dim // len(node_types))] = 1.0
            embeddings[node_type] = embedding
        
        return embeddings
    
    def encode_node(self, node_data: Dict) -> torch.Tensor:
        """Encode a node into a feature vector"""
        # Start with node type embedding
        node_type = node_data.get('node_type', 'disease')
        features = self.node_type_embeddings.get(node_type, self.node_type_embeddings['disease']).clone()
        
        # Add confidence score if available
        if 'confidence' in node_data:
            confidence = node_data['confidence']
            features[-10:] = torch.tensor([confidence] * 10)
        
        # Add other attributes as features
        if 'attributes' in node_data:
            attrs = node_data['attributes']
            if 'severity' in attrs:
                severity_map = {'mild': 0.25, 'moderate': 0.5, 'severe': 0.75, 'critical': 1.0}
                severity_val = severity_map.get(attrs['severity'], 0.5)
                features[-20:-10] = torch.tensor([severity_val] * 10)
        
        return features

class EdgeEncoder:
    """Encode edges into feature vectors"""
    
    def __init__(self):
        self.relation_types = [
            'has_symptom', 'treated_by', 'prevented_by', 'lifestyle_factor', 
            'nutrition_factor', 'disease_symptom', 'disease_drug', 'disease_prevention'
        ]
    
    def encode_edge(self, edge_data: Dict) -> torch.Tensor:
        """Encode an edge into a feature vector"""
        # One-hot encoding for relation type
        relation_type = edge_data.get('relation_type', '')
        relation_idx = self.relation_types.index(relation_type) if relation_type in self.relation_types else 0
        
        features = torch.zeros(len(self.relation_types))
        features[relation_idx] = 1.0
        
        # Add weight and confidence
        weight = edge_data.get('weight', 1.0)
        confidence = edge_data.get('confidence', 1.0)
        
        features = torch.cat([features, torch.tensor([weight, confidence])])
        
        return features

# Example usage
if __name__ == "__main__":
    # Initialize fusion system
    fusion_system = KnowledgeGraphFusion(
        node_feature_dim=128,
        hidden_dim=64,
        output_dim=32,
        n_heads=4,
        n_layers=2
    )
    
    # Create sample knowledge graphs
    from knowledge_graph import HealthcareKnowledgeGraph
    from llm_extractor import HealthcareRelation, RelationType
    
    # Primary graph
    primary_kg = HealthcareKnowledgeGraph()
    sample_relations = [
        HealthcareRelation("Diabetes", "High Blood Sugar", RelationType.DISEASE_SYMPTOM, 0.9, "Primary symptom"),
        HealthcareRelation("Diabetes", "Insulin", RelationType.DISEASE_DRUG, 0.95, "Primary treatment")
    ]
    primary_kg.build_from_relations(sample_relations)
    
    # Fuse graphs
    fused_kg = fusion_system.fuse_graphs(primary_kg)
    
    print(f"Fused graph has {fused_kg.graph.number_of_nodes()} nodes and {fused_kg.graph.number_of_edges()} edges")
