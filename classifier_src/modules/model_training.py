import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from sklearn.utils import resample
from typing import Tuple, Dict, Optional


def create_balanced_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    negative_to_positive_ratio: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a balanced training set by undersampling the majority (legitimate) class.
    Keeps all minority (laundering) samples and randomly samples majority class.

    Args:
        X_train: Training features
        y_train: Training labels
        negative_to_positive_ratio: Ratio of legitimate to laundering transactions
                                     (e.g., 10 means 10 legitimate for every 1 laundering)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (balanced X_train, balanced y_train)
    """
    print("=" * 80)
    print("CREATING BALANCED TRAINING SET")
    print("=" * 80)

    # Separate positive and negative samples
    positive_mask = y_train == 1
    negative_mask = y_train == 0

    X_positive = X_train[positive_mask]
    y_positive = y_train[positive_mask]

    X_negative = X_train[negative_mask]
    y_negative = y_train[negative_mask]

    n_positive = len(y_positive)
    n_negative = len(y_negative)

    print(f"\nOriginal Training Set:")
    print(f"  Laundering (positive): {n_positive:,} ({n_positive/len(y_train)*100:.3f}%)")
    print(f"  Legitimate (negative): {n_negative:,} ({n_negative/len(y_train)*100:.3f}%)")
    print(f"  Imbalance ratio: 1:{n_negative/n_positive:.0f}")

    # Calculate target number of negative samples
    n_negative_target = n_positive * negative_to_positive_ratio

    if n_negative_target > n_negative:
        print(f"\nWarning: Requested ratio 1:{negative_to_positive_ratio} requires {n_negative_target:,} negative samples")
        print(f"         but only {n_negative:,} available. Using all negative samples.")
        n_negative_target = n_negative

    # Undersample majority class
    indices = resample(
        np.arange(len(y_negative)),
        n_samples=n_negative_target,
        random_state=random_state,
        replace=False
    )

    X_negative_sampled = X_negative[indices]
    y_negative_sampled = y_negative[indices]

    # Combine positive and sampled negative
    X_balanced = np.vstack([X_positive, X_negative_sampled])
    y_balanced = np.hstack([y_positive, y_negative_sampled])

    # Shuffle
    shuffle_indices = np.random.RandomState(random_state).permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    print(f"\nBalanced Training Set:")
    print(f"  Laundering (positive): {(y_balanced == 1).sum():,} ({(y_balanced == 1).mean()*100:.2f}%)")
    print(f"  Legitimate (negative): {(y_balanced == 0).sum():,} ({(y_balanced == 0).mean()*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{(y_balanced == 0).sum()/(y_balanced == 1).sum():.0f}")
    print(f"  Total samples: {len(y_balanced):,} (reduced from {len(y_train):,})")
    print(f"  Reduction: {(1 - len(y_balanced)/len(y_train))*100:.1f}%")
    print("=" * 80 + "\n")

    return X_balanced, y_balanced


def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_split: float = 0.1,
    **xgb_params
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost classifier with automatic class weight calculation and early stopping.

    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        validation_split: Fraction of training data to use for validation
        **xgb_params: Additional XGBoost parameters to override defaults

    Returns:
        Tuple of (trained model, training history dict)
    """
    # Split off validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
    )

    # Calculate class imbalance
    n_negative = (y_tr == 0).sum()
    n_positive = (y_tr == 1).sum()
    scale_pos_weight = n_negative / n_positive

    print("=" * 80)
    print("TRAINING XGBOOST CLASSIFIER")
    print("=" * 80)
    print(f"\nTraining Set:")
    print(f"  Total samples: {len(y_tr):,}")
    print(f"  Legitimate: {n_negative:,} ({n_negative/len(y_tr)*100:.2f}%)")
    print(f"  Laundering: {n_positive:,} ({n_positive/len(y_tr)*100:.2f}%)")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    print(f"\nValidation Set:")
    print(f"  Total samples: {len(y_val):,}")
    print(f"  Legitimate: {(y_val == 0).sum():,}")
    print(f"  Laundering: {(y_val == 1).sum():,}")

    # Default parameters
    default_params = {
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'aucpr',
        'early_stopping_rounds': 10
    }

    # Override with user-provided params
    default_params.update(xgb_params)

    print(f"\nModel Parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")

    # Train model
    print(f"\nTraining...")
    model = xgb.XGBClassifier(**default_params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Get training history
    history = {
        'validation_score': model.evals_result()['validation_0']['aucpr'],
        'best_iteration': model.best_iteration,
        'best_score': model.best_score
    }

    print(f"\nTraining complete!")
    print(f"  Best iteration: {history['best_iteration']}")
    print(f"  Best validation PR-AUC: {history['best_score']:.4f}")
    print("=" * 80 + "\n")

    return model, history


def evaluate_classifier(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list = ['Legitimate', 'Laundering']
) -> Dict:
    """
    Evaluate trained classifier on test set and print comprehensive metrics.

    Args:
        model: Trained XGBoost classifier
        X_test: Test features
        y_test: Test labels
        class_names: Names of classes for display

    Returns:
        Dictionary containing all computed metrics
    """
    print("=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Calculate rates from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)  # False positive rate
    fnr = fn / (fn + tp)  # False negative rate

    # Print results
    print(f"\nTest Set Size: {len(y_test):,}")
    print(f"  {class_names[0]}: {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"  {class_names[1]}: {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.2f}%)")

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    {class_names[0]:>12s}  {class_names[1]:>12s}")
    print(f"  Actual {class_names[0]:>10s}  {tn:>12,}  {fp:>12,}   ({tn/(tn+fp)*100:.1f}% correct)")
    print(f"         {class_names[1]:>10s}  {fn:>12,}  {tp:>12,}   ({tp/(tp+fn)*100:.1f}% correct)")

    print(f"\nClassification Metrics:")
    print(f"  Precision:      {precision:.4f}  (of predicted {class_names[1].lower()}, % actually {class_names[1].lower()})")
    print(f"  Recall:         {recall:.4f}  (of actual {class_names[1].lower()}, % detected)")
    print(f"  F1 Score:       {f1:.4f}")
    print(f"  ROC-AUC:        {roc_auc:.4f}")
    print(f"  PR-AUC:         {pr_auc:.4f} (best metric for imbalanced data)")

    print(f"\nError Rates:")
    print(f"  False Positive Rate: {fpr:.4f}  ({fp:,} legitimate flagged as {class_names[1].lower()})")
    print(f"  False Negative Rate: {fnr:.4f}  ({fn:,} {class_names[1].lower()} missed)")

    print("=" * 80 + "\n")

    # Return metrics dictionary
    metrics = {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'fnr': fnr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return metrics


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list,
    top_k: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> pd.DataFrame:
    """
    Plot feature importance from trained XGBoost model.

    Args:
        model: Trained XGBoost classifier
        feature_names: List of feature names
        top_k: Number of top features to display
        figsize: Figure size

    Returns:
        DataFrame with feature importance scores
    """
    # Get feature importance
    importance = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Select top K
    top_features = importance_df.head(top_k)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_k} Most Important Features', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    print(f"Top {top_k} Features:")
    print(importance_df.head(top_k).to_string(index=False))
    print()

    return importance_df


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array
        class_names: Names of classes
        normalize: If True, show percentages instead of counts
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )

    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()


def compare_model_performance(
    baseline_metrics: Dict,
    xgboost_metrics: Dict
) -> pd.DataFrame:
    """
    Compare performance between baseline (UMAP 2D) and XGBoost (full features).

    Args:
        baseline_metrics: Metrics from baseline model (e.g., UMAP 2D classifier)
        xgboost_metrics: Metrics from XGBoost classifier

    Returns:
        DataFrame with side-by-side comparison
    """
    print("=" * 80)
    print("MODEL COMPARISON: UMAP 2D Logistic Regression vs XGBoost (58D)")
    print("=" * 80)

    metrics_to_compare = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']

    comparison_data = []
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0)
        xgb_val = xgboost_metrics.get(metric, 0)
        improvement = ((xgb_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0

        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'UMAP 2D': f"{baseline_val:.4f}",
            'XGBoost 58D': f"{xgb_val:.4f}",
            'Improvement': f"{improvement:+.1f}%"
        })

    comparison_df = pd.DataFrame(comparison_data)

    print(f"\n{comparison_df.to_string(index=False)}")

    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    xgb_f1 = xgboost_metrics['f1_score']
    baseline_f1 = baseline_metrics['f1_score']

    if xgb_f1 > baseline_f1 * 1.1:
        print("✓ Significant improvement: XGBoost leverages full feature space effectively")
    elif xgb_f1 > baseline_f1 * 1.02:
        print("~ Moderate improvement: Full features help, but gains are modest")
    else:
        print("⚠ Minimal improvement: 2D UMAP captures most discriminative information")

    print(f"\nXGBoost achieves F1={xgb_f1:.3f}, PR-AUC={xgboost_metrics['pr_auc']:.3f}")
    print("=" * 80 + "\n")

    return comparison_df
