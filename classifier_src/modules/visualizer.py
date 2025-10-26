import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def create_transaction_embeddings(trans_df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Create transaction embeddings from engineered features.
    Concatenates normalized numerical features with one-hot encoded categorical features.
    Excludes account IDs (from_account_id, to_account_id) as they represent graph structure.

    Args:
        trans_df: DataFrame with all engineered features

    Returns:
        Tuple of:
        - embeddings: numpy array of shape (n_transactions, n_features)
        - feature_names: list of feature names corresponding to embedding columns
    """
    # Numerical features to include
    numerical_cols = [
        'amount_paid', 'amount_received',
        'hour_sin', 'hour_cos', 'time_normalized',
        'from_tx_rate', 'from_avg_amount', 'from_std_amount', 'from_unique_destinations',
        'from_destination_diversity', 'from_days_active', 'from_is_first_tx',
        'to_tx_rate', 'to_avg_amount', 'to_std_amount', 'to_unique_sources',
        'to_source_diversity', 'to_days_active', 'to_is_first_tx',
        'amount_zscore_from', 'amount_zscore_to'
    ]

    # Categorical features to one-hot encode
    categorical_cols = ['payment_currency_id', 'receiving_currency_id', 'payment_format_id']

    # Extract numerical features
    numerical_features = trans_df[numerical_cols].values

    # One-hot encode categorical features
    categorical_encoded = pd.get_dummies(trans_df[categorical_cols], columns=categorical_cols, prefix=categorical_cols)

    # Concatenate numerical and categorical features
    embeddings = np.concatenate([numerical_features, categorical_encoded.values], axis=1)

    # Create feature names list
    feature_names = numerical_cols + categorical_encoded.columns.tolist()

    return embeddings, feature_names


def stratified_sample_for_visualization(
    trans_df: pd.DataFrame,
    legitimate_sample_size: int = 5000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample for UMAP visualization.
    Includes all laundering transactions and a random sample of legitimate transactions.

    Args:
        trans_df: DataFrame with is_laundering column
        legitimate_sample_size: Number of legitimate transactions to sample
        random_state: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    # Get all laundering transactions
    laundering_df = trans_df[trans_df['is_laundering'] == 1].copy()

    # Sample legitimate transactions
    legitimate_df = trans_df[trans_df['is_laundering'] == 0].sample(
        n=min(legitimate_sample_size, (trans_df['is_laundering'] == 0).sum()),
        random_state=random_state
    )

    # Combine and shuffle
    sample_df = pd.concat([laundering_df, legitimate_df], ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Stratified Sample Created:")
    print(f"  Total transactions: {len(sample_df):,}")
    print(f"  Laundering: {(sample_df['is_laundering'] == 1).sum():,} ({(sample_df['is_laundering'] == 1).mean()*100:.2f}%)")
    print(f"  Legitimate: {(sample_df['is_laundering'] == 0).sum():,} ({(sample_df['is_laundering'] == 0).mean()*100:.2f}%)")

    return sample_df


def apply_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction to embeddings.

    Args:
        embeddings: Input embeddings array of shape (n_samples, n_features)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        random_state: Random seed for reproducibility

    Returns:
        2D UMAP coordinates of shape (n_samples, 2)
    """
    print(f"Applying UMAP reduction...")
    print(f"  Input shape: {embeddings.shape}")
    print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=True
    )

    umap_coords = reducer.fit_transform(embeddings)

    print(f"  Output shape: {umap_coords.shape}")
    print(f"UMAP reduction complete.")

    return umap_coords


def plot_umap_embeddings(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Visualization of Transaction Embeddings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot UMAP 2D embeddings with color-coding for laundering vs legitimate transactions.

    Args:
        umap_coords: 2D UMAP coordinates of shape (n_samples, 2)
        labels: Binary labels (0 = legitimate, 1 = laundering)
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Separate legitimate and laundering transactions
    legitimate_mask = labels == 0
    laundering_mask = labels == 1

    # Plot legitimate transactions (smaller, semi-transparent)
    ax.scatter(
        umap_coords[legitimate_mask, 0],
        umap_coords[legitimate_mask, 1],
        c='blue',
        alpha=0.3,
        s=10,
        label=f'Legitimate ({legitimate_mask.sum():,}, {legitimate_mask.mean()*100:.2f}%)',
        edgecolors='none'
    )

    # Plot laundering transactions (larger, opaque)
    ax.scatter(
        umap_coords[laundering_mask, 0],
        umap_coords[laundering_mask, 1],
        c='red',
        alpha=0.8,
        s=50,
        label=f'Laundering ({laundering_mask.sum():,}, {laundering_mask.mean()*100:.2f}%)',
        edgecolors='black',
        linewidths=0.5
    )

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def compute_umap_separation_metrics(
    umap_coords: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    Compute separation metrics to quantify how well UMAP separates laundering from legitimate transactions.

    Args:
        umap_coords: 2D UMAP coordinates of shape (n_samples, 2)
        labels: Binary labels (0 = legitimate, 1 = laundering)

    Returns:
        Dictionary containing all computed metrics
    """
    print("=" * 80)
    print("UMAP SEPARATION DIAGNOSTICS")
    print("=" * 80)

    metrics = {}

    # 1. Clustering Quality Metrics
    print("\n1. CLUSTERING QUALITY METRICS")
    print("-" * 80)

    silhouette = silhouette_score(umap_coords, labels)
    davies_bouldin = davies_bouldin_score(umap_coords, labels)
    calinski_harabasz = calinski_harabasz_score(umap_coords, labels)

    metrics['silhouette_score'] = silhouette
    metrics['davies_bouldin_index'] = davies_bouldin
    metrics['calinski_harabasz_score'] = calinski_harabasz

    print(f"  Silhouette Score:        {silhouette:.4f}  (range: [-1, 1], higher = better separation)")
    print(f"  Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower = better separation)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f}  (higher = better separation)")

    # 2. Distance-Based Metrics
    print("\n2. DISTANCE-BASED METRICS")
    print("-" * 80)

    legitimate_mask = labels == 0
    laundering_mask = labels == 1

    legitimate_coords = umap_coords[legitimate_mask]
    laundering_coords = umap_coords[laundering_mask]

    legitimate_centroid = legitimate_coords.mean(axis=0)
    laundering_centroid = laundering_coords.mean(axis=0)

    centroid_distance = np.linalg.norm(legitimate_centroid - laundering_centroid)

    within_legitimate = np.mean([np.linalg.norm(point - legitimate_centroid) for point in legitimate_coords])
    within_laundering = np.mean([np.linalg.norm(point - laundering_centroid) for point in laundering_coords])
    avg_within_class = (within_legitimate + within_laundering) / 2

    separation_ratio = centroid_distance / avg_within_class if avg_within_class > 0 else 0

    metrics['centroid_distance'] = centroid_distance
    metrics['avg_within_class_distance'] = avg_within_class
    metrics['separation_ratio'] = separation_ratio

    print(f"  Centroid Distance:           {centroid_distance:.4f}")
    print(f"  Avg Within-Class Distance:   {avg_within_class:.4f}")
    print(f"  Separation Ratio:            {separation_ratio:.4f}  (higher = better, >1 means clusters are separated)")

    # 3. 2D Classification Performance
    print("\n3. 2D CLASSIFICATION PERFORMANCE (Logistic Regression on UMAP coords)")
    print("-" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        umap_coords, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    metrics['confusion_matrix'] = cm
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    metrics['roc_auc'] = roc_auc
    metrics['pr_auc'] = pr_auc

    print(f"  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"     [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
    print(f"\n  Precision:  {precision:.4f}  (of predicted laundering, % actually laundering)")
    print(f"  Recall:     {recall:.4f}  (of actual laundering, % detected)")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}  (precision-recall AUC, good for imbalanced data)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if silhouette > 0.3 and separation_ratio > 1.5:
        print("✓ Good separation: Laundering and legitimate transactions form distinct clusters")
    elif silhouette > 0.1 and separation_ratio > 1.0:
        print("~ Moderate separation: Some clustering visible, but with overlap")
    else:
        print("✗ Poor separation: Classes are not well-separated in UMAP space")

    print(f"\nA simple 2D classifier achieves F1={f1:.3f}, PR-AUC={pr_auc:.3f}")
    print("=" * 80 + "\n")

    return metrics
