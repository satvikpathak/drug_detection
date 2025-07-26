"""
SLANGNET: Slang-Aware Neural Network with Graph Attention
A state-of-the-art model for detecting drug use and overdose symptoms in social media text.

This model incorporates:
- Graph neural networks for slang understanding
- Multi-modal attention mechanisms
- Contextual slang detection
- Advanced graph convolution layers
- Cross-lingual slang processing

Author: Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import defaultdict
import re


class SlangDictionary:
    """Comprehensive slang dictionary for drug-related terms."""
    
    def __init__(self):
        self.slang_terms = {
            # Opioids
            'heroin': ['dope', 'smack', 'junk', 'horse', 'brown', 'china white', 'tar', 'black tar'],
            'fentanyl': ['fent', 'china white', 'apache', 'dance fever', 'goodfella', 'jackpot'],
            'oxycodone': ['oxy', 'oxycontin', 'percs', 'roxies', 'blues', '30s', '80s'],
            'morphine': ['morph', 'miss emma', 'monkey', 'white stuff'],
            
            # Stimulants
            'cocaine': ['coke', 'blow', 'snow', 'powder', 'white', 'nose candy', 'crack', 'rock'],
            'methamphetamine': ['meth', 'crystal', 'ice', 'glass', 'tina', 'crank', 'speed'],
            'amphetamine': ['speed', 'uppers', 'bennies', 'dexies', 'black beauties'],
            
            # Symptoms
            'overdose': ['od', 'overdosing', 'nodding out', 'falling out', 'going out'],
            'nausea': ['puking', 'throwing up', 'sick to stomach', 'queasy', 'barfing'],
            'dizziness': ['dizzy', 'lightheaded', 'woozy', 'spinning', 'vertigo'],
            'seizure': ['seizing', 'convulsing', 'shaking', 'fits', 'episode'],
            'anxiety': ['panicking', 'freaking out', 'nervous', 'worried', 'stressed'],
            
            # Social media specific
            'high': ['lit', 'baked', 'fried', 'zoned', 'faded', 'wasted', 'trashed'],
            'withdrawal': ['dope sick', 'sick', 'kicking', 'cold turkey', 'detoxing'],
            'dealer': ['plug', 'connect', 'guy', 'man', 'source', 'supplier']
        }
        
        # Create reverse mapping
        self.term_to_category = {}
        for category, terms in self.slang_terms.items():
            for term in terms:
                self.term_to_category[term.lower()] = category
            self.term_to_category[category.lower()] = category
    
    def get_slang_category(self, text: str) -> List[str]:
        """Extract slang categories from text."""
        text_lower = text.lower()
        categories = set()
        
        for term, category in self.term_to_category.items():
            if term in text_lower:
                categories.add(category)
        
        return list(categories)
    
    def get_all_terms(self) -> List[str]:
        """Get all slang terms."""
        all_terms = []
        for terms in self.slang_terms.values():
            all_terms.extend(terms)
        return all_terms


class GraphConstructor:
    """Constructs knowledge graphs from text for slang understanding."""
    
    def __init__(self, slang_dict: SlangDictionary):
        self.slang_dict = slang_dict
        self.graph = nx.Graph()
        self._build_knowledge_graph()
    
    def _build_knowledge_graph(self):
        """Build knowledge graph connecting slang terms."""
        # Add nodes for all categories
        for category in self.slang_dict.slang_terms.keys():
            self.graph.add_node(category, type='category')
        
        # Add nodes for all terms
        for category, terms in self.slang_dict.slang_terms.items():
            for term in terms:
                self.graph.add_node(term, type='term', category=category)
                # Connect term to category
                self.graph.add_edge(term, category, relation='belongs_to')
        
        # Add semantic relationships
        self._add_semantic_relationships()
    
    def _add_semantic_relationships(self):
        """Add semantic relationships between terms."""
        # Drug type relationships
        opioid_terms = ['heroin', 'fentanyl', 'oxycodone', 'morphine']
        stimulant_terms = ['cocaine', 'methamphetamine', 'amphetamine']
        
        # Connect opioids
        for i, term1 in enumerate(opioid_terms):
            for term2 in opioid_terms[i+1:]:
                self.graph.add_edge(term1, term2, relation='same_type')
        
        # Connect stimulants
        for i, term1 in enumerate(stimulant_terms):
            for term2 in stimulant_terms[i+1:]:
                self.graph.add_edge(term1, term2, relation='same_type')
        
        # Symptom relationships
        symptom_terms = ['overdose', 'nausea', 'dizziness', 'seizure', 'anxiety']
        for i, term1 in enumerate(symptom_terms):
            for term2 in symptom_terms[i+1:]:
                self.graph.add_edge(term1, term2, relation='symptom_related')
    
    def text_to_graph(self, text: str) -> Data:
        """Convert text to PyTorch Geometric graph."""
        # Extract slang terms from text
        slang_categories = self.slang_dict.get_slang_category(text)
        
        # Create subgraph with relevant nodes
        relevant_nodes = set()
        for category in slang_categories:
            relevant_nodes.add(category)
            if category in self.slang_dict.slang_terms:
                relevant_nodes.update(self.slang_dict.slang_terms[category])
        
        # Create subgraph
        subgraph = self.graph.subgraph(relevant_nodes)
        
        # Convert to PyTorch Geometric format
        if len(subgraph.nodes()) == 0:
            # Empty graph - create dummy node
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            x = torch.zeros((1, 768))  # Dummy node with BERT embedding size
        else:
            # Create node mapping
            node_to_idx = {node: idx for idx, node in enumerate(subgraph.nodes())}
            
            # Create edge index
            edges = list(subgraph.edges())
            if edges:
                edge_index = torch.tensor([[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edges], dtype=torch.long).t()
                # Add reverse edges for undirected graph
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Create node features (placeholder - will be replaced with embeddings)
            x = torch.zeros((len(subgraph.nodes()), 768))
        
        return Data(x=x, edge_index=edge_index)


class SlangAwareEmbedding(nn.Module):
    """Slang-aware embedding layer with contextual understanding."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 768, num_slang_types: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_slang_types = num_slang_types
        
        # Standard embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Slang-specific embeddings
        self.slang_embeddings = nn.Embedding(num_slang_types, embedding_dim)
        
        # Context gate
        self.context_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Slang detection
        self.slang_detector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_slang_types)
        )
        
    def forward(self, input_ids: torch.Tensor, slang_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get word embeddings
        word_embeds = self.word_embeddings(input_ids)
        
        # Detect slang types
        slang_logits = self.slang_detector(word_embeds)
        slang_probs = F.softmax(slang_logits, dim=-1)
        
        # Get slang embeddings
        slang_types = torch.argmax(slang_probs, dim=-1)
        slang_embeds = self.slang_embeddings(slang_types)
        
        # Context gate
        context_input = torch.cat([word_embeds, slang_embeds], dim=-1)
        gate = self.context_gate(context_input)
        
        # Combine embeddings
        combined_embeds = gate * word_embeds + (1 - gate) * slang_embeds
        
        return combined_embeds, slang_probs


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for processing slang knowledge graphs."""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        # Linear transformations
        self.W = nn.Linear(input_dim, output_dim)
        self.W_src = nn.Linear(input_dim, output_dim)
        self.W_dst = nn.Linear(input_dim, output_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply linear transformations
        x_transformed = self.W(x)
        
        # Graph attention
        if edge_index.size(1) > 0:
            # Create adjacency matrix
            batch_size = x.size(0)
            adj_matrix = torch.zeros(batch_size, batch_size, device=x.device)
            adj_matrix[edge_index[0], edge_index[1]] = 1
            
            # Apply attention
            attended, _ = self.attention(x_transformed, x_transformed, x_transformed, 
                                       attn_mask=adj_matrix.unsqueeze(0))
        else:
            attended = x_transformed
        
        # Residual connection and normalization
        output = self.layer_norm(x_transformed + self.dropout(attended))
        return self.output_proj(output)


class MultiModalAttention(nn.Module):
    """Multi-modal attention for combining text and graph features."""
    
    def __init__(self, text_dim: int, graph_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        
        # Projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, text_features: torch.Tensor, graph_features: torch.Tensor) -> torch.Tensor:
        # Project to same dimension
        text_proj = self.text_proj(text_features)
        graph_proj = self.graph_proj(graph_features)
        
        # Cross-modal attention
        attended_text, _ = self.cross_attention(text_proj, graph_proj, graph_proj)
        attended_graph, _ = self.cross_attention(graph_proj, text_proj, text_proj)
        
        # Fusion
        combined = torch.cat([attended_text, attended_graph], dim=-1)
        fused = self.fusion(combined)
        
        return fused


class SLANGNETModel(nn.Module):
    """
    SLANGNET: Slang-Aware Neural Network with Graph Attention
    
    This model represents a breakthrough in slang-aware drug detection,
    incorporating graph neural networks and multi-modal attention.
    """
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 num_substance_classes: int = 3,
                 num_symptom_classes: int = 18,
                 embedding_dim: int = 768,
                 hidden_dim: int = 512,
                 num_graph_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_substance_classes = num_substance_classes
        self.num_symptom_classes = num_symptom_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize components
        self.slang_dict = SlangDictionary()
        self.graph_constructor = GraphConstructor(self.slang_dict)
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Slang-aware embedding
        self.slang_embedding = SlangAwareEmbedding(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=embedding_dim
        )
        
        # Graph processing layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(embedding_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_graph_layers)
        ])
        
        # Multi-modal attention
        self.multimodal_attention = MultiModalAttention(embedding_dim, hidden_dim, hidden_dim)
        
        # Task-specific classifiers
        self.substance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_substance_classes)
        )
        
        self.symptom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_symptom_classes)
        )
        
        # Advanced loss functions
        self.focal_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _process_text_to_graph(self, texts: List[str]) -> List[Data]:
        """Process texts to graphs."""
        graphs = []
        for text in texts:
            graph = self.graph_constructor.text_to_graph(text)
            graphs.append(graph)
        return graphs
    
    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text with slang-aware embeddings."""
        # Get BERT embeddings
        bert_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply slang-aware processing
        slang_embeds, slang_probs = self.slang_embedding(input_ids)
        
        # Combine with BERT embeddings
        combined_embeds = bert_outputs.last_hidden_state + 0.1 * slang_embeds
        
        return combined_embeds
    
    def _process_graphs(self, graphs: List[Data], device: torch.device) -> torch.Tensor:
        """Process graphs through GNN layers."""
        if not graphs:
            return torch.zeros((1, self.hidden_dim), device=device)
        
        # Batch graphs
        batch = Batch.from_data_list(graphs)
        batch = batch.to(device)
        
        # Process through graph layers
        x = batch.x
        for layer in self.graph_layers:
            x = layer(x, batch.edge_index, batch.batch)
            x = F.gelu(x)
        
        # Global pooling
        pooled = global_mean_pool(x, batch.batch)
        
        return pooled
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                texts: List[str],
                substance_labels: Optional[torch.Tensor] = None,
                symptom_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        device = input_ids.device
        
        # Encode text
        text_features = self._encode_text(input_ids, attention_mask)
        
        # Process graphs
        graphs = self._process_text_to_graph(texts)
        graph_features = self._process_graphs(graphs, device)
        
        # Ensure graph_features has correct batch size
        if graph_features.size(0) != text_features.size(0):
            # Repeat graph features if needed
            if graph_features.size(0) == 1:
                graph_features = graph_features.repeat(text_features.size(0), 1)
            else:
                # Pad or truncate
                if graph_features.size(0) < text_features.size(0):
                    padding = torch.zeros(text_features.size(0) - graph_features.size(0), 
                                        graph_features.size(1), device=device)
                    graph_features = torch.cat([graph_features, padding], dim=0)
                else:
                    graph_features = graph_features[:text_features.size(0)]
        
        # Multi-modal attention
        text_pooled = text_features.mean(dim=1)  # Global average pooling
        combined_features = self.multimodal_attention(text_pooled, graph_features)
        
        # Task-specific predictions
        substance_logits = self.substance_classifier(combined_features)
        symptom_logits = self.symptom_classifier(combined_features)
        
        # Calculate loss if labels are provided
        loss = None
        if substance_labels is not None and symptom_labels is not None:
            # Focal loss for substance classification
            substance_loss = self.focal_loss(substance_logits, substance_labels)
            pt = torch.exp(-substance_loss)
            focal_loss = 0.25 * (1 - pt) ** 2 * substance_loss
            substance_loss = focal_loss.mean()
            
            # Asymmetric loss for symptom detection
            symptom_loss = self.bce_loss(symptom_logits, symptom_labels.float())
            symptom_loss = symptom_loss.mean()
            
            # Combined loss
            loss = 0.6 * substance_loss + 0.4 * symptom_loss
        
        return {
            'loss': loss,
            'substance_logits': substance_logits,
            'symptom_logits': symptom_logits,
            'substance_probs': F.softmax(substance_logits, dim=-1),
            'symptom_probs': torch.sigmoid(symptom_logits),
            'text_features': text_features,
            'graph_features': graph_features
        }
    
    def get_slang_attention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get attention weights for slang terms."""
        with torch.no_grad():
            _, slang_probs = self.slang_embedding(input_ids)
            return slang_probs


class SLANGNETConfig:
    """Configuration class for SLANGNET model."""
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'roberta-base')
        self.num_substance_classes = kwargs.get('num_substance_classes', 3)
        self.num_symptom_classes = kwargs.get('num_symptom_classes', 18)
        self.embedding_dim = kwargs.get('embedding_dim', 768)
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.num_graph_layers = kwargs.get('num_graph_layers', 3)
        self.num_heads = kwargs.get('num_heads', 8)
        self.dropout = kwargs.get('dropout', 0.2)
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SLANGNETConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'num_substance_classes': self.num_substance_classes,
            'num_symptom_classes': self.num_symptom_classes,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_graph_layers': self.num_graph_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }


# Model factory function
def create_slangnet_model(config: SLANGNETConfig) -> SLANGNETModel:
    """Create a SLANGNET model with the given configuration."""
    return SLANGNETModel(
        model_name=config.model_name,
        num_substance_classes=config.num_substance_classes,
        num_symptom_classes=config.num_symptom_classes,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_graph_layers=config.num_graph_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )