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

    graph.ndata['node_feats'] = node_features

    return graph

class EdgeFeatsEmbedding(nn.Module):
    def __init__(
            self, 
            num_currencies=15, 
            num_payment_methods=7, 
            currencies_embed_dim=8, 
            payment_embed_dim=8, 
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
    
class MLP_classifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_layers=3):
        super(MLP_classifier, self).__init__()
        FC_layers = [nn.Linear(input_dim, input_dim) for _ in range(hidden_layers)]
        FC_layers.append(nn.Linear(input_dim, output_dim)) # output layer project to no. classes
        self.FC_layers = nn.ModuleList(FC_layers)
        self.hidden_layers = hidden_layers
    
    def forward(self, x):
        y = x
        for layer in range(self.hidden_layers):
            y = self.FC_layers[layer](y)
            y = torch.relu(y)
        y = self.FC_layers[self.hidden_layers](y) #output layer hiddendim --> 2 classes
        return y

class graph_MHA_layer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8):
        super().__init__()
        self.per_head_hidden_dim = hidden_dim//num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale_factor = torch.sqrt(torch.tensor(self.per_head_hidden_dim))
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)

    def message_func(self, edges):
        qikj = (edges.dst['Q'] * edges.src['K']).sum(dim=2).unsqueeze(2)
        scores = qikj / self.scale_factor
        vj = edges.src['V']
        return {'scores' : scores, 'vj' : vj}
    
    def reduce_func(self, nodes):
        scores = nodes.mailbox['scores']
        vj = nodes.mailbox['vj']
        max_scores = scores.max(dim=1, keepdim=True)[0]
        stable_exp = torch.exp(scores - max_scores)
        attn_weights = stable_exp / stable_exp.sum(dim=1, keepdim=True)
        h = torch.sum(attn_weights * vj, dim=1)
        return {'h' : h}
    
    def forward(self, g, h):
        Q = self.WQ(h)
        K = self.WK(h)
        V = self.WV(h)
        g.ndata['Q'] = Q.reshape(-1, self.num_heads, self.per_head_hidden_dim)
        g.ndata['K'] = K.reshape(-1, self.num_heads, self.per_head_hidden_dim)
        g.ndata['V'] = V.reshape(-1, self.num_heads, self.per_head_hidden_dim)

        g.update_all(self.message_func, self.reduce_func)

        gMHA = g.ndata['h']
        return gMHA

class GraphTransformer_layer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_mha = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)
        self.gMHA = graph_MHA_layer(hidden_dim, num_heads)
        self.WO = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, g, h):
        h_residual = h
        h = self.layer_norm1(h)
        h_MHA = self.gMHA(g, h).reshape(-1, self.hidden_dim)
        h_MHA = self.dropout_mha(h_MHA)
        h_MHA = self.WO(h_MHA)
        h = h_residual + h_MHA

        h_residual = h
        h = self.layer_norm2(h)
        h_MLP = self.linear1(h)
        h_MLP = torch.relu(h_MLP)
        h_MLP = self.dropout_mlp(h_MLP)
        h_MLP = self.linear2(h_MLP)
        h = h_residual + h_MLP

        return h

class GraphTransformer_net(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, transformer_layers=7):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.GraphTransformer_layers = nn.ModuleList([GraphTransformer_layer(hidden_dim=hidden_dim) for _ in range(transformer_layers)])
        self.embedding_edge_feats = EdgeFeatsEmbedding()
        self.MLP_classifier = MLP_classifier(hidden_dim + hidden_dim + self.embedding_edge_feats.out_dim, output_dim)

    def forward(self, g):
        h = g.ndata['node_feats']
        h = self.embedding_h(h)

        for GT_layer in self.GraphTransformer_layers:
            h = GT_layer(g, h)
        
        edge_feats_embed = self.embedding_edge_feats(g.edata['payment_currency'], g.edata['receiving_currency'], g.edata['payment_format'], g.edata['numericals']) # (E, 21)
        src_nodes, dst_nodes = g.edges()
        src_embed = h[src_nodes] # (E, 128)
        dst_embed = h[dst_nodes] # (E, 128)

        edge_embed = torch.cat([src_embed, dst_embed, edge_feats_embed], dim=1) # (E, 277) each edge has 277 dim representation made up of its source node, dest node and transaction features
        y = self.MLP_classifier(edge_embed)
        return y


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

def main(output_dir='./saved_models/', num_epochs=100, learning_rate=0.001):
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
    model = GraphTransformer_net().to(DEVICE)

    train_labels = training_graph.edata['is_laundering'].long()

    ############ MINORITY CLASS SCALE WEIGHT ############
    class_weights = torch.tensor([1.0, 400.0]).to(DEVICE) #place 30x weight on positive examples
    ######################################################

    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # support code for loss level undersampling
    train_labels = training_graph.edata['is_laundering'].long()
    class_0_idx = (train_labels == 0).nonzero(as_tuple=True)[0]
    class_1_idx = (train_labels == 1).nonzero(as_tuple=True)[0]
    n_class_1 = len(class_1_idx)
    n_class_0_sample = n_class_1 * 500 # We want to sample 100 legitimate transactions for every laundering transaction during loss calc

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
        
        # Evaluate every 1 epochs, can change to 5 if uw, this slows down training 
        if epoch % 1 == 0:
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
                torch.save(model.state_dict(), output_dir + 'best_transformer_model.pt')
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
    model.load_state_dict(torch.load(output_dir + 'best_transformer_model.pt'))
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