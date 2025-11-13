# Import dependencies
from modules.data_loader import *
from modules.feature_engineering import *
from modules.visualizer import *
import networkx as nx
import dgl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
import kagglehub
import argparse

DATASET_NAME = "HI-Small"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_graph_from_df(df):
    source_accounts = df['from_account_id'].values
    dest_accounts = df['to_account_id'].values

    graph = dgl.graph((source_accounts, dest_accounts))
    print(f"Number of nodes: {graph.num_nodes()}")
    print(f"Number of edges: {graph.num_edges()}")
    # EDGE FEATS

    # bookkeeping
    graph.edata['transaction_id'] = torch.tensor(df['transaction_id'].values, dtype=torch.long)
    graph.edata['is_laundering'] = torch.tensor(df['is_laundering'].values, dtype=torch.long) #not used in training

    # numericals: we can include them directly
    numericals = ['amount_received', 'amount_paid', 'hour_sin', 'hour_cos', 'time_normalized']
    edge_numerical = torch.tensor(df[numericals].values, dtype=torch.float32)
    graph.edata['numericals'] = edge_numerical

    # categoricals: we use learned embeddings. This makes them transductive, but that is ok because currencies and payment types don't change often

    payment_currency = torch.tensor(df['payment_currency_id'].values, dtype=torch.long)
    receiving_currency = torch.tensor(df['receiving_currency_id'].values, dtype=torch.long)
    payment_format = torch.tensor(df['payment_format_id'].values, dtype=torch.long)
    graph.edata['payment_currency'] = payment_currency
    graph.edata['receiving_currency'] = receiving_currency
    graph.edata['payment_format'] = payment_format

    # NODE FEATS

    # For each account, we compute their in and outdegrees
    # We zero-center and log transform this value and normalize by the graph's average transformed in and outdegrees so that our model is not sensitive to the graph size. Our training graph has 80% of the edges, so naturally the raw degree counts will be higher than the testing graph.
    indegrees = graph.in_degrees().float()
    outdegrees = graph.out_degrees().float()

    log_indegrees = torch.log(indegrees + 1)
    log_outdegrees = torch.log(outdegrees + 1)

    avg_log_indegree = log_indegrees.mean()
    avg_log_outdegree = log_outdegrees.mean()

    std_log_indegree = log_indegrees.std()
    std_log_outdegree = log_outdegrees.std()

    # zc and normalize
    normalized_indegree = (log_indegrees - avg_log_indegree) / std_log_indegree
    normalized_outdegree = (log_outdegrees - avg_log_outdegree) / std_log_outdegree

    node_features = torch.stack([normalized_indegree, normalized_outdegree], dim=1)

    # Perhaps we can include simple graph RWPE as a node feature too?
    graph.ndata['node_feats'] = node_features

    return graph

class EdgeEmbedding(nn.Module):
    def __init__(
            self, 
            num_currencies=15, 
            num_payment_methods=7, 
            currencies_embed_dim=8, 
            payment_embed_dim=4, 
            num_numericals=5):
        super().__init__()
        self.currencies_embed = nn.Embedding(num_currencies, currencies_embed_dim)
        self.payment_embed = nn.Embedding(num_payment_methods, payment_embed_dim)

        # each edge has payment currency, receiving currency, payment method and numericals
        self.out_dim = currencies_embed_dim + currencies_embed_dim + payment_embed_dim + num_numericals

    def forward(self, payment_curr, receiving_curr, payment_method, numericals):
        payment_curr_embed = self.currencies_embed(payment_curr)
        receiving_curr_embed = self.currencies_embed(receiving_curr)
        payment_method_embed = self.payment_embed(payment_method)

        edge_feats = torch.cat([payment_curr_embed, receiving_curr_embed, payment_method_embed, numericals], dim=1)
        return edge_feats
    
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, residual=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.residual = residual

        if residual and input_dim != output_dim:
            self.res_linear = nn.Linear(input_dim, output_dim) # project residual if diff dims
        elif residual:
            self.res_linear = nn.Identity()

    def message_func(self, edges):
        return {'m': edges.src['Wh'] * edges.src['norm']} # normalize by source degree first

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}
    
    def forward(self, graph, node_feats):
        with graph.local_scope():
            h_in = node_feats

            graph.ndata['Wh'] = self.linear(node_feats) # W is linear transform, h is node feats

            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(1)
            graph.ndata['norm'] = norm

            graph.update_all(self.message_func, self.reduce_func)

            h_out = graph.ndata['h'] * norm # then normalize by destination degree

            h_out = F.relu(h_out)
            if self.residual:
                return h_out + self.res_linear(h_in)
            else:
                return h_out

class GCNEdgeClassifier(nn.Module):
    
    def __init__(self,
                # node params
                node_in_feats=2, # log transformed and normalized indegree outdegree
                hidden_dim=64,
                num_gcn_layers=3,

                # for EdgeEmbedding
                num_currencies=15,
                num_payment_methods=7,
                currencies_embed_dim=8,
                payment_embed_dim=4,
                num_numericals=5,

                # output classes
                num_classes=2,

                # regularization
                dropout=0.2,
                use_batch_norm=False):
        super().__init__()

        self.num_gcn_layers = num_gcn_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(node_in_feats, hidden_dim, residual=False)) # no residual for layer 1 since in dim and out dim dont match, simpler to just skip
        for i in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, residual=True))
        if use_batch_norm: # create batch norm layers if using them
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(num_gcn_layers)])

        # Edge Feature processor
        self.edge_embed = EdgeEmbedding(num_currencies, num_payment_methods, currencies_embed_dim, payment_embed_dim, num_numericals)

        # Combiner to do edge classification: append 2 node embeddings with edge embedding
        edge_repr_dim = 2*hidden_dim + self.edge_embed.out_dim

        # MLP for edge classification
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_repr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, graph):
        node_features = graph.ndata['node_feats']
        payment_currency = graph.edata['payment_currency']
        receiving_currency = graph.edata['receiving_currency']
        payment_method = graph.edata['payment_format']
        edge_numericals = graph.edata['numericals']

        # GCN message passing to learn node embeddings
        h = node_features
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(graph, h)
            if self.use_batch_norm:
                h = self.batch_norms[i](h)
            if i < self.num_gcn_layers - 1: # dropout between each layer except after last
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # get node embedding for each edge
        src_nodes, dst_nodes = graph.edges()
        src_embed = h[src_nodes]
        dst_embed = h[dst_nodes]

        # process edge features
        edge_features = self.edge_embed(payment_currency, receiving_currency, payment_method, edge_numericals)

        # concat 2 nodes embeddings with edge features to get complete edge embedding
        edge_repr = torch.cat([src_embed, dst_embed, edge_features], dim=1)

        # classify edges
        logits = self.edge_classifier(edge_repr)

        return logits
    
def evaluate(model, graph):
    """Evaluate model performance."""
    model.eval()
    
    with torch.no_grad():
        # Pass edge features as separate arguments from graph.edata
        logits = model(graph)
        
        labels = graph.edata['is_laundering']
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        accuracy = (preds == labels).float().mean().item()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels.cpu().numpy(), 
            preds.cpu().numpy(),
            average=None,
            zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
        except:
            auc = 0.0
        
        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'auc': auc,
            'confusion_matrix': cm
        }
        
        return metrics, preds, probs

def main(output_dir='./saved_models/', num_epochs=50, learning_rate=0.001):
    """
    Main function to pre-process data, train/evaluate the model.
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    dataset_name = "HI-Small"

    print(f"Loading {dataset_name}...\n")
    trans_df = load_transactions(dataset_size=DATASET_NAME)

    # Apply pre-processing / feature engineering steps

    # 1. convert all currencies to USD for normaliztion
    trans_df = convert_currency_to_USD(trans_df)

    # 2. compute sinusoidal temporal encodings and normalized unix timestamp
    trans_df = temporal_encoding(trans_df)

    # 3. give each currency and payment method a unique integer ID
    trans_df = encode_currency_ids(trans_df)
    trans_df = encode_payment_format_ids(trans_df)

    # 4. give each account a unique integer ID
    trans_df, account_to_id, id_to_account = encode_account_ids(trans_df)

    trans_df = normalize_amounts(trans_df)

    # Apply temporal train/test split
    # split data temporally: first 80% for training, last 20% for testing
    train_df, test_df = temporal_train_test_split(trans_df, train_ratio=0.8)

    # Build graphs
    training_graph = build_graph_from_df(train_df)
    test_graph = build_graph_from_df(test_df)

    # Setup model and loss
    training_graph = training_graph.to(DEVICE)
    test_graph = test_graph.to(DEVICE)
    model = GCNEdgeClassifier().to(DEVICE)

    train_labels = training_graph.edata['is_laundering'].long()

    ############ MINORITY CLASS SCALE WEIGHT ############
    class_weights = torch.tensor([1.0, 40.0]).to(DEVICE) #place 20x weight on positive examples, since we are doing 1:30 undersampling, feel free to adjust these numbers and experiment yourself
    ######################################################

    print(f"Class weights: {class_weights}")
    # criterion = FocalLoss(alpha=class_weights, gamma=1.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # support code for loss level undersampling
    train_labels = training_graph.edata['is_laundering'].long()
    class_0_idx = (train_labels == 0).nonzero(as_tuple=True)[0]
    class_1_idx = (train_labels == 1).nonzero(as_tuple=True)[0]
    n_class_1 = len(class_1_idx)
    n_class_0_sample = n_class_1 * 30 # We want to sample 30 legitimate transactions for every laundering transaction during loss calc

    best_f1 = 0.0

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    for epoch in range(num_epochs):

        # perform the random subsampling of legitimate transactions every epoch
        class_0_sampled = class_0_idx[torch.randperm(len(class_0_idx))[:n_class_0_sample]]
        train_idx = torch.cat([class_0_sampled, class_1_idx])
        train_mask = torch.zeros(len(train_labels), dtype=torch.bool)
        train_mask[train_idx] = True

        # Train
        model.train()
        optimizer.zero_grad()
        
        logits = model(training_graph)
        loss = criterion(logits[train_mask], train_labels[train_mask]) # apply the mask that subsamples legitimate transactions when calculating loss
        
        loss.backward()
        optimizer.step()
        
        # Evaluate every 2 epochs, can change to 5 if uw, this slows down training 
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                eval_logits = model(training_graph)
                preds = eval_logits.argmax(dim=1)
                
                acc = (preds == train_labels).float().mean().item()
                
                # Class 1 metrics
                tp = ((preds == 1) & (train_labels == 1)).sum().item()
                fp = ((preds == 1) & (train_labels == 0)).sum().item()
                fn = ((preds == 0) & (train_labels == 1)).sum().item()
                
                precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
                
                pred_1_pct = (preds == 1).float().mean().item() * 100
                
            print(f"\nEpoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
            print(f"  Class 1 - P: {precision_1:.4f}, R: {recall_1:.4f}, F1: {f1_1:.4f}")
            print(f"  Predicting class 1: {pred_1_pct:.5f}%")
            
            # Save best model
            if f1_1 > best_f1:
                best_f1 = f1_1
                torch.save(model.state_dict(), output_dir + 'best_gcn_model.pt')
                print(f"  Saved best model (F1: {best_f1:.4f})")
        else:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    test_labels = test_graph.edata['is_laundering'].long()

    print(f"Class 0: {(test_labels==0).sum()} ({100*(test_labels==0).float().mean():.2f}%)")
    print(f"Class 1: {(test_labels==1).sum()} ({100*(test_labels==1).float().mean():.2f}%)")

    # Evaluate
    model.load_state_dict(torch.load(output_dir + 'best_gcn_model.pt'))
    model.eval()
    with torch.no_grad():
        test_logits = model(test_graph)
        test_preds = test_logits.argmax(dim=1)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1]
        
        # Metrics
        acc = (test_preds == test_labels).float().mean().item()
        
        tp = ((test_preds == 1) & (test_labels == 1)).sum().item()
        fp = ((test_preds == 1) & (test_labels == 0)).sum().item()
        fn = ((test_preds == 0) & (test_labels == 1)).sum().item()
        tn = ((test_preds == 0) & (test_labels == 0)).sum().item()
        
        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
        
        try:
            auc = roc_auc_score(test_labels.cpu().numpy(), test_probs.cpu().numpy())
        except:
            auc = 0.0

    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:7d}  FP: {fp:7d}")
    print(f"  FN: {fn:7d}  TP: {tp:7d}")
    print(f"\nClass 0 (Legitimate):")
    print(f"  Precision: {precision_0:.4f}")
    print(f"  Recall: {recall_0:.4f}")
    print(f"\nClass 1 (Laundering):")
    print(f"  Precision: {precision_1:.4f}")
    print(f"  Recall: {recall_1:.4f}")
    print(f"  F1 Score: {f1_1:.4f}")
    print("="*60)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GNN Model training on IBM AML Transactions dataset')
    parser.add_argument('--output_dir', type=str, default='./saved_models/',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    print("Preparing dataset")
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    print("Path to dataset files:", path)
    print("Begin main loop")
    
    main(args.output_dir, args.num_epochs, args.learning_rate)