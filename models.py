
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv, GraphormerLayer
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm

class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for edge classification in AML detection.
    
    Uses multi-head attention and incorporates edge features.
    """
    
    def __init__(
        self,
        in_node_feats: int,
        in_edge_feats: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_heads: List[int] = [8, 8, 1],
        num_classes: int = 2,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        residual: bool = True
    ):
        """
        Initialize GAT model.
        
        Args:
            in_node_feats: Input node feature dimension
            in_edge_feats: Input edge feature dimension
            hidden_dims: Hidden dimensions for each GAT layer
            num_heads: Number of attention heads for each layer
            num_classes: Number of output classes
            dropout: Dropout rate
            negative_slope: LeakyReLU negative slope
            residual: Whether to use residual connections
        """
        super(GraphAttentionNetwork, self).__init__()
        
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(in_edge_feats, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Input layer
        self.gat_layers.append(
            GATConv(
                in_node_feats,
                hidden_dims[0] // num_heads[0],
                num_heads[0],
                feat_drop=dropout,
                attn_drop=dropout,
                negative_slope=negative_slope,
                residual=residual,
                activation=F.elu
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            in_dim = hidden_dims[i-1]
            out_dim = hidden_dims[i] // num_heads[i]
            
            self.gat_layers.append(
                GATConv(
                    in_dim,
                    out_dim,
                    num_heads[i],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=F.elu if i < len(hidden_dims) - 1 else None
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dims[i]))
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2 + hidden_dims[0], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, 
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            graph: DGL graph
            node_features: Node feature tensor [num_nodes, in_node_feats]
            edge_features: Edge feature tensor [num_edges, in_edge_feats]
            
        Returns:
            Edge predictions [num_edges, num_classes]
        """
        h = node_features
        
        # Process through GAT layers
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h = gat_layer(graph, h)
            h = h.flatten(1) if h.dim() == 3 else h  # Flatten multi-head outputs
            h = norm(h)
            if i < len(self.gat_layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Project edge features
        edge_feat = self.edge_proj(edge_features)
        
        # Get edge representations (concatenate source, dest, and edge features)
        src, dst = graph.edges()
        edge_repr = torch.cat([h[src], h[dst], edge_feat], dim=1)
        
        # Classify edges
        out = self.edge_classifier(edge_repr)
        
        return out
    
class AMLModelTrainer:
    """
    Trainer for AML detection models with comprehensive evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: GNN model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Class weights for imbalanced data
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.history = {
            'train_loss': [], 'train_f1': [],
            'val_loss': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [], 'val_auc': []
        }
        
    def train_epoch(self, graph: dgl.DGLGraph, mask: torch.Tensor) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            graph: DGL graph
            mask: Training mask for edges
            
        Returns:
            Tuple of (loss, f1_score)
        """
        self.model.train()
        graph = graph.to(self.device)
        
        # Forward pass
        logits = self.model(
            graph,
            graph.ndata['feat'],
            graph.edata['feat']
        )
        
        # Apply mask
        logits_masked = logits[mask]
        labels_masked = graph.edata['label'][mask]
        
        # Compute loss
        loss = self.criterion(logits_masked, labels_masked)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute F1 score
        with torch.no_grad():
            preds = logits_masked.argmax(dim=1)
            f1 = f1_score(
                labels_masked.cpu().numpy(),
                preds.cpu().numpy(),
                average='macro'
            )
        
        return loss.item(), f1
    
    @torch.no_grad()
    def evaluate(self, graph: dgl.DGLGraph, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            graph: DGL graph
            mask: Evaluation mask for edges
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        graph = graph.to(self.device)
        
        # Forward pass
        logits = self.model(
            graph,
            graph.ndata['feat'],
            graph.edata['feat']
        )
        
        # Apply mask
        logits_masked = logits[mask]
        labels_masked = graph.edata['label'][mask]
        
        # Compute loss
        loss = self.criterion(logits_masked, labels_masked)
        
        # Predictions
        preds = logits_masked.argmax(dim=1).cpu().numpy()
        probs = F.softmax(logits_masked, dim=1)[:, 1].cpu().numpy()
        labels = labels_masked.cpu().numpy()
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'f1_binary': f1_score(labels, preds, average='binary'),
            'precision': precision_score(labels, preds, average='macro', zero_division=0),
            'recall': recall_score(labels, preds, average='macro', zero_division=0),
            'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def train(
        self,
        graph: dgl.DGLGraph,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train model with early stopping.
        
        Args:
            graph: DGL graph with train/val/test masks
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_f1 = 0
        patience_counter = 0
        
        train_mask = graph.edata['train_mask']
        val_mask = graph.edata['val_mask']
        
        if verbose:
            print("\n" + "="*70)
            print(f"Training {self.model.__class__.__name__}")
            print("="*70)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_f1 = self.train_epoch(graph, train_mask)
            
            # Validate
            val_metrics = self.evaluate(graph, val_mask)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1_macro'])
            
            epoch_time = time.time() - start_time
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Time: {epoch_time:.2f}s | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train F1: {train_f1:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val F1: {val_metrics['f1_macro']:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            print("\n" + "="*70)
            print(f"Training completed. Best Val F1: {best_val_f1:.4f}")
            print("="*70 + "\n")
        
        return self.history
    
    def test(self, graph: dgl.DGLGraph, verbose: bool = True) -> Dict[str, float]:
        """
        Test model on test set.
        
        Args:
            graph: DGL graph with test mask
            verbose: Whether to print results
            
        Returns:
            Test metrics
        """
        test_mask = graph.edata['test_mask']
        test_metrics = self.evaluate(graph, test_mask)
        
        if verbose:
            print("\n" + "="*70)
            print("Test Results")
            print("="*70)
            print(f"Test Loss:      {test_metrics['loss']:.4f}")
            print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
            print(f"Test F1 (Binary): {test_metrics['f1_binary']:.4f}")
            print(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
            print(f"Test Precision: {test_metrics['precision']:.4f}")
            print(f"Test Recall:    {test_metrics['recall']:.4f}")
            print(f"Test AUC:       {test_metrics['auc']:.4f}")
            print("\nConfusion Matrix:")
            print(test_metrics['confusion_matrix'])
            print("="*70 + "\n")
        
        return test_metrics