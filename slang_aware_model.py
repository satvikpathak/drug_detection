import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

class SlangAwareAttention(nn.Module):
    """
    Slang-Aware Attention Layer for social media text processing.
    Incorporates emoji embeddings and slang-specific gating mechanisms.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Slang-aware gating mechanism
        self.slang_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Emoji embeddings for common emojis
        self.emoji_embedding = nn.Embedding(100, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, emoji_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Slang-Aware Attention Layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            emoji_ids: Optional emoji IDs tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply slang-aware gating
        slang_gate_weights = self.slang_gate(x)
        x_gated = x * slang_gate_weights
        
        # Project to hidden dimension
        x_projected = nn.Linear(self.input_dim, self.hidden_dim).to(x.device)(x_gated)
        
        # Add emoji embeddings if available
        if emoji_ids is not None:
            emoji_emb = self.emoji_embedding(emoji_ids)
            x_projected = x_projected + emoji_emb
        
        # Multi-head attention
        attn_output, _ = self.attention(
            x_projected, x_projected, x_projected,
            attn_mask=attention_mask
        )
        
        # Residual connection and layer norm
        x_residual = self.layer_norm1(x_projected + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x_residual)
        
        # Final residual connection and layer norm
        output = self.layer_norm2(x_residual + self.dropout(ff_output))
        
        return output

class ATTENDModel(nn.Module):
    """
    ATTEND (Attention-based Text Drug Event Neural Detection) Model.
    Multi-task learning framework for drug use and overdose symptom detection.
    """
    
    def __init__(self, input_size: int, num_substance_classes: int, num_symptom_labels: int,
                 hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_substance_classes = num_substance_classes
        self.num_symptom_labels = num_symptom_labels
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_dim)
        
        # Slang-aware attention layers
        self.attention_layers = nn.ModuleList([
            SlangAwareAttention(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Substance classification head
        self.substance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_substance_classes)
        )
        
        # Symptom detection head
        self.symptom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_symptom_labels),
            nn.Sigmoid()  # Multi-label classification
        )
        
        # Feature importance layer
        self.feature_importance = nn.Linear(hidden_dim, input_size)
        
    def forward(self, x: torch.Tensor, emoji_ids: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass of the ATTEND model.
        
        Args:
            x: Input TF-IDF features of shape (batch_size, input_size)
            emoji_ids: Optional emoji IDs
            
        Returns:
            Dictionary containing predictions and attention weights
        """
        batch_size = x.shape[0]
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # Reshape for attention layers (treat as sequence of length 1)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply attention layers
        attention_weights = []
        for attention_layer in self.attention_layers:
            x = attention_layer(x, emoji_ids)
            attention_weights.append(x)
        
        # Global pooling
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # Substance classification
        substance_logits = self.substance_classifier(x)
        substance_probs = F.softmax(substance_logits, dim=1)
        
        # Symptom detection
        symptom_probs = self.symptom_classifier(x)
        
        # Feature importance
        feature_importance = torch.sigmoid(self.feature_importance(x))
        
        return {
            'substance_logits': substance_logits,
            'substance_probs': substance_probs,
            'symptom_probs': symptom_probs,
            'feature_importance': feature_importance,
            'attention_weights': attention_weights
        }

class WeightedLoss(nn.Module):
    """
    Weighted loss function for multi-task learning.
    """
    
    def __init__(self, alpha: float = 0.6, substance_loss_weight: float = 1.0, 
                 symptom_loss_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.substance_loss_weight = substance_loss_weight
        self.symptom_loss_weight = symptom_loss_weight
        
        self.substance_loss = nn.CrossEntropyLoss()
        self.symptom_loss = nn.BCELoss()
        
    def forward(self, substance_logits: torch.Tensor, substance_labels: torch.Tensor,
                symptom_probs: torch.Tensor, symptom_labels: torch.Tensor) -> dict:
        """
        Compute weighted loss for both tasks.
        
        Args:
            substance_logits: Substance classification logits
            substance_labels: Substance ground truth labels
            symptom_probs: Symptom detection probabilities
            symptom_labels: Symptom ground truth labels
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Substance classification loss
        substance_loss = self.substance_loss(substance_logits, substance_labels)
        
        # Symptom detection loss
        symptom_loss = self.symptom_loss(symptom_probs, symptom_labels.float())
        
        # Weighted combination
        total_loss = (self.alpha * self.substance_loss_weight * substance_loss + 
                     (1 - self.alpha) * self.symptom_loss_weight * symptom_loss)
        
        return {
            'total_loss': total_loss,
            'substance_loss': substance_loss,
            'symptom_loss': symptom_loss
        }

def extract_emoji_ids(text: str) -> List[int]:
    """
    Extract emoji IDs from text for emoji embedding.
    This is a simplified version - in practice, you'd use a proper emoji library.
    """
    emoji_mapping = {
        '😊': 0, '😢': 1, '😡': 2, '😴': 3, '😵': 4, '🤢': 5, '🤮': 6,
        '💊': 7, '💉': 8, '🚬': 9, '🍺': 10, '🥃': 11, '💊': 12
    }
    
    emoji_ids = []
    for char in text:
        if char in emoji_mapping:
            emoji_ids.append(emoji_mapping[char])
        else:
            emoji_ids.append(0)  # No emoji
    
    return emoji_ids[:len(text)]  # Truncate to text length

def create_social_media_text(text: str) -> str:
    """
    Transform formal text to social media style.
    """
    import random
    
    # Add common social media patterns
    slang_mappings = {
        'very': 'v',
        'really': 'rly',
        'because': 'bc',
        'before': 'b4',
        'through': 'thru',
        'you': 'u',
        'are': 'r',
        'your': 'ur',
        'for': '4',
        'to': '2',
        'too': '2',
        'two': '2'
    }
    
    # Apply slang mappings
    for formal, slang in slang_mappings.items():
        text = text.replace(f' {formal} ', f' {slang} ')
    
    # Add hashtags for drug-related terms
    drug_terms = ['drug', 'medication', 'pill', 'overdose', 'symptom', 'pain']
    for term in drug_terms:
        if term in text.lower():
            text += f' #{term}'
    
    # Add emojis randomly
    emojis = ['😊', '😢', '😡', '😴', '😵', '🤢', '🤮', '💊', '💉']
    if random.random() < 0.3:  # 30% chance to add emoji
        text += f' {random.choice(emojis)}'
    
    return text

def evaluate_model_performance(model: ATTENDModel, test_loader, device: str) -> dict:
    """
    Comprehensive model evaluation with detailed metrics.
    """
    model.eval()
    all_substance_preds = []
    all_substance_labels = []
    all_symptom_preds = []
    all_symptom_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, substance_labels, symptom_labels = batch
            features = features.to(device)
            substance_labels = substance_labels.to(device)
            symptom_labels = symptom_labels.to(device)
            
            outputs = model(features)
            
            # Substance predictions
            substance_preds = torch.argmax(outputs['substance_probs'], dim=1)
            all_substance_preds.extend(substance_preds.cpu().numpy())
            all_substance_labels.extend(substance_labels.cpu().numpy())
            
            # Symptom predictions
            symptom_preds = (outputs['symptom_probs'] > 0.5).float()
            all_symptom_preds.extend(symptom_preds.cpu().numpy())
            all_symptom_labels.extend(symptom_labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    substance_accuracy = accuracy_score(all_substance_labels, all_substance_preds)
    substance_f1 = f1_score(all_substance_labels, all_substance_preds, average='weighted')
    
    symptom_f1 = f1_score(all_symptom_labels, all_symptom_preds, average='micro')
    symptom_precision = precision_score(all_symptom_labels, all_symptom_preds, average='micro')
    symptom_recall = recall_score(all_symptom_labels, all_symptom_preds, average='micro')
    
    return {
        'substance_accuracy': substance_accuracy,
        'substance_f1': substance_f1,
        'symptom_f1': symptom_f1,
        'symptom_precision': symptom_precision,
        'symptom_recall': symptom_recall,
        'substance_predictions': all_substance_preds,
        'substance_labels': all_substance_labels,
        'symptom_predictions': all_symptom_preds,
        'symptom_labels': all_symptom_labels
    }