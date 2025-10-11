"""
Visualization utilities for EDA.
Provides plotting functions for transaction network analysis.

Creation of the visualization functionality used in this module was heavily AI assisted (syntax and debugging help, information and suggestions on correct arguments and features of diagrams that can be included).

Model used was Claude Sonnet 4.5 (model ID: claude-sonnet-4-5-20250929), both on claude.ai web interface and vscode Claude Code extension.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List
import plotly.graph_objects as go
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")


def set_publication_style():
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

# Not currently used u can add this method to main notebook to save the figs you want
def save_figure(fig, filename: str, output_dir: str = 'outputs/figures'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filepath}")


# =============================================================================
# Transaction-level visualizations for notebook section 3
# =============================================================================

def plot_transaction_volume_over_time( # 3.1 Temporal Patterns
    transactions_df: pd.DataFrame,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(14, 6))

    trans_counts = transactions_df.set_index('timestamp').resample('H').size()

    ax.plot(trans_counts.index, trans_counts.values, linewidth=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Transactions')
    ax.set_title(f'Transaction Volume Over Time (Hourly granularity)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_amount_distribution( # 3.2 Amount Analysis
    transactions_df: pd.DataFrame
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    legit = transactions_df[transactions_df['is_laundering'] == 0]['amount_received']
    launder = transactions_df[transactions_df['is_laundering'] == 1]['amount_received']

    #left histogram
    axes[0].hist(legit, bins=50, alpha=0.6, label='Legitimate', log=True)
    axes[0].hist(launder, bins=50, alpha=0.6, label='Laundering', log=True)
    axes[0].set_xlabel('Amount (log scale)')
    axes[0].set_ylabel('Frequency (log scale)')
    axes[0].set_xscale('log')
    axes[0].set_title('Transaction Amount Distribution')
    axes[0].legend()

    # right box plot
    data_to_plot = [legit, launder]
    axes[1].boxplot(data_to_plot, labels=['Legitimate', 'Laundering'], vert=True)
    axes[1].set_ylabel('Amount')
    axes[1].set_yscale('log')
    axes[1].set_title('Amount Comparison')


    plt.tight_layout()
    return fig

def plot_payment_type_breakdown( # 3.2 Amount Analysis
    transactions_df: pd.DataFrame,
) -> plt.Figure:
    
    payment_breakdown = transactions_df.groupby(['payment_format', 'is_laundering']).size().unstack(fill_value=0)
    payment_breakdown.columns = ['Legitimate', 'Laundering']
    payment_breakdown['total'] = payment_breakdown.sum(axis=1)
    payment_breakdown['laundering_pct'] = (payment_breakdown['Laundering'] / payment_breakdown['total'] * 100)
    payment_breakdown = payment_breakdown.sort_values('total', ascending=False)

    dataset_avg_rate = transactions_df['is_laundering'].sum() / len(transactions_df) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # left stacked bar, although you cant really see the red parts
    ax1 = axes[0]
    payment_breakdown[['Legitimate', 'Laundering']].plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=['#4472C4', '#C5504B'],
        alpha=0.8
    )
    ax1.set_xlabel('Payment Type')
    ax1.set_ylabel('Number of Transactions')
    ax1.set_title('Payment Type Distribution (Legitimate vs Laundering)')
    ax1.legend(['Legitimate', 'Laundering'], loc='upper right')
    ax1.set_xticklabels(payment_breakdown.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for i, payment_type in enumerate(payment_breakdown.index):
        total = payment_breakdown.loc[payment_type, 'total']
        pct = payment_breakdown.loc[payment_type, 'laundering_pct']
        if pct > 0:
            ax1.text(i, total, f'{pct:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # right horizontal bar
    ax2 = axes[1]
    payment_breakdown['laundering_pct'].plot(
        kind='barh',
        ax=ax2,
        color=['#C5504B' if x > dataset_avg_rate else '#4472C4' for x in payment_breakdown['laundering_pct']],
        alpha=0.8
    )
    ax2.set_xlabel('Laundering Rate (%)')
    ax2.set_ylabel('Payment Type')
    ax2.set_title('Laundering Rate by Payment Type')
    ax2.axvline(x=dataset_avg_rate, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Dataset Avg ({dataset_avg_rate:.3f}%)')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    for i, (payment_type, row) in enumerate(payment_breakdown.iterrows()):
        pct = row['laundering_pct']
        ax2.text(pct, i, f' {pct:.3f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig

# =============================================================================
# Account level visualizations for notebook section 4
# =============================================================================

def compute_account_activity_statistics(account_stats_df: pd.DataFrame) -> pd.DataFrame: # sec 4.1
    """
    Compute detailed statistics about account activity distribution.

    Includes percentile breakdowns and laundering rates by activity tier.

    Args:
        account_stats_df: DataFrame with 'total_transactions' and laundering indicators

    Returns:
        DataFrame with statistics by activity tier
    """
    total_trans = account_stats_df['total_transactions']

    percentiles = [0, 25, 50, 75, 90, 95, 99, 99.9, 100]
    tier_stats = []

    has_laundering = any(col in account_stats_df.columns for col in ['laundering_sent', 'laundering_received'])

    for i in range(len(percentiles) - 1):
        lower_pct = percentiles[i]
        upper_pct = percentiles[i + 1]

        lower_val = total_trans.quantile(lower_pct / 100)
        upper_val = total_trans.quantile(upper_pct / 100)

        tier_mask = (total_trans >= lower_val) & (total_trans < upper_val)
        if i == len(percentiles) - 2:
            tier_mask = (total_trans >= lower_val) & (total_trans <= upper_val)

        tier_accounts = account_stats_df[tier_mask]

        tier_info = {
            'tier': f'P{lower_pct}-P{upper_pct}',
            'tx_range': f'{lower_val:.0f} - {upper_val:.0f}',
            'num_accounts': len(tier_accounts),
            'pct_of_total': len(tier_accounts) / len(account_stats_df) * 100,
            'median_tx': tier_accounts['total_transactions'].median(),
        }

        if has_laundering:
            laundering_mask = (tier_accounts.get('laundering_sent', 0) > 0) | (tier_accounts.get('laundering_received', 0) > 0)
            num_laundering = laundering_mask.sum()
            tier_info['laundering_accounts'] = num_laundering
            tier_info['laundering_rate_pct'] = num_laundering / len(tier_accounts) * 100 if len(tier_accounts) > 0 else 0

        tier_stats.append(tier_info)

    return pd.DataFrame(tier_stats)

def plot_transaction_frequency_per_account(account_stats_df: pd.DataFrame) -> plt.Figure: # sec 4.1
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    axes[3].remove()

    total_trans = account_stats_df['total_transactions']

    has_laundering = any(col in account_stats_df.columns for col in ['laundering_sent', 'laundering_received'])
    if has_laundering:
        laundering_mask = (account_stats_df.get('laundering_sent', 0) > 0) | (account_stats_df.get('laundering_received', 0) > 0)
        legit_trans = total_trans[~laundering_mask]
        launder_trans = total_trans[laundering_mask]

    # top left, logscale histogram of legit vs laundering
    ax1 = axes[0]
    if has_laundering:
        ax1.hist(legit_trans, bins=50, edgecolor='black', alpha=0.6, label='Legitimate', color='steelblue')
        ax1.hist(launder_trans, bins=50, edgecolor='black', alpha=0.6, label='Laundering-Involved', color='crimson')
        ax1.legend()
    else:
        ax1.hist(total_trans, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

    ax1.set_xlabel('Total Transactions per Account (log scale)')
    ax1.set_ylabel('Number of Accounts (log scale)')
    ax1.set_title('Full Distribution (Log-Log Scale)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax1.axvline(total_trans.median(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {total_trans.median():.0f}')
    ax1.axvline(total_trans.quantile(0.95), color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'95th: {total_trans.quantile(0.95):.0f}')

    # top right, 0-95th percentile histogram
    ax2 = axes[1]
    p95 = total_trans.quantile(0.95)

    if has_laundering:
        legit_zoom = legit_trans[legit_trans <= p95]
        launder_zoom = launder_trans[launder_trans <= p95]
        ax2.hist(legit_zoom, bins=30, edgecolor='black', alpha=0.6, label='Legitimate', color='steelblue')
        ax2.hist(launder_zoom, bins=30, edgecolor='black', alpha=0.6, label='Laundering-Involved', color='crimson')
        ax2.legend()
    else:
        trans_zoom = total_trans[total_trans <= p95]
        ax2.hist(trans_zoom, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

    ax2.set_xlabel('Total Transactions per Account')
    ax2.set_ylabel('Number of Accounts')
    ax2.set_title(f'Zoomed View (0 - 95th Percentile: {p95:.0f})')
    ax2.grid(axis='y', alpha=0.3)

    ax2.axvline(total_trans.median(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {total_trans.median():.0f}')
    ax2.axvline(total_trans.quantile(0.25), color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'Q1: {total_trans.quantile(0.25):.0f}')
    ax2.axvline(total_trans.quantile(0.75), color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'Q3: {total_trans.quantile(0.75):.0f}')
    ax2.legend(fontsize=8)

    # bottom left, empirical Cumulative Distribution Function
    ax3 = axes[2]

    if has_laundering:
        legit_sorted = np.sort(legit_trans)
        legit_ecdf = np.arange(1, len(legit_sorted) + 1) / len(legit_sorted) * 100
        ax3.plot(legit_sorted, legit_ecdf, label='Legitimate', linewidth=2, color='steelblue')

        if len(launder_trans) > 0:
            launder_sorted = np.sort(launder_trans)
            launder_ecdf = np.arange(1, len(launder_sorted) + 1) / len(launder_sorted) * 100
            ax3.plot(launder_sorted, launder_ecdf, label='Laundering-Involved', linewidth=2, color='crimson')
    else:
        sorted_trans = np.sort(total_trans)
        ecdf = np.arange(1, len(sorted_trans) + 1) / len(sorted_trans) * 100
        ax3.plot(sorted_trans, ecdf, linewidth=2, color='steelblue')

    ax3.set_xlabel('Total Transactions per Account')
    ax3.set_ylabel('Cumulative Percentage (%)')
    ax3.set_title('Cumulative Distribution (ECDF)')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    for pct, label in [(50, '50%'), (95, '95%')]:
        ax3.axhline(pct, color='gray', linestyle='--', alpha=0.3)
        ax3.text(total_trans.min(), pct, f' {label}', va='center', fontsize=8, color='gray')

    plt.tight_layout()
    return fig

# =============================================================================
# Pattern analysis visualizations for section 6
# =============================================================================

def plot_pattern_type_distribution(pattern_df: pd.DataFrame) -> plt.Figure: #sec 6.1
    fig, ax = plt.subplots(figsize=(12, 6))

    pattern_counts = pattern_df.groupby('pattern_id')['pattern_type'].first().value_counts()

    colors = plt.cm.Set3(range(len(pattern_counts)))
    ax.bar(range(len(pattern_counts)), pattern_counts.values, color=colors)
    ax.set_xticks(range(len(pattern_counts)))
    ax.set_xticklabels(pattern_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Patterns')
    ax.set_title('Laundering Pattern Type Distribution')
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(pattern_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    return fig


def plot_pattern_temporal_clustering( #sec 6.2
    pattern_df: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))

    pattern_times = pattern_df.groupby('pattern_id').agg({
        'timestamp': 'min',
        'pattern_type': 'first'
    }).reset_index()

    pattern_types = pattern_times['pattern_type'].unique()
    type_to_num = {pt: i for i, pt in enumerate(pattern_types)}
    pattern_times['pattern_num'] = pattern_times['pattern_type'].map(type_to_num)

    for ptype in pattern_types:
        mask = pattern_times['pattern_type'] == ptype
        ax.scatter(
            pattern_times[mask]['timestamp'],
            pattern_times[mask]['pattern_num'],
            label=ptype,
            alpha=0.6,
            s=50
        )

    ax.set_xlabel('Time')
    ax.set_ylabel('Pattern Type')
    ax.set_yticks(range(len(pattern_types)))
    ax.set_yticklabels(pattern_types)
    ax.set_title('Temporal Distribution of Laundering Patterns')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_ego_network( #sec 6.3
    subgraph: nx.MultiDiGraph,
    center_node: str,
    layout: str = 'spring',
    figsize=(12, 12)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    if layout == 'spring':
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(subgraph)
    else:
        pos = nx.kamada_kawai_layout(subgraph)

    node_colors = []
    for node in subgraph.nodes():
        if node == center_node:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')

    edge_colors = []
    
    for u, v, data in subgraph.edges(data=True):
        if data.get('is_laundering', 0) == 1:
            edge_colors.append('red')
        else:
            edge_colors.append('gray')

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=300, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, arrows=True, ax=ax, alpha=0.5, arrowsize=10)

    labels = {center_node: center_node}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)

    ax.set_title(f'Ego Network for Account {center_node}')
    ax.axis('off')

    plt.tight_layout()
    return fig

# =============================================================================
# Temporal analysis visualizations for section 7
# =============================================================================

def plot_hourly_distribution(transactions_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    hour_of_day = transactions_df['timestamp'].dt.hour
    hour_counts = hour_of_day.value_counts().sort_index()

    ax.bar(range(24), [hour_counts.get(h, 0) for h in range(0,24)])
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Transactions')
    ax.set_title('Transaction Distribution by Hour of Day')
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_daily_distribution(transactions_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    day_of_week = transactions_df['timestamp'].dt.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_counts = day_of_week.value_counts().sort_index()

    ax.bar(day_names, [day_counts.get(d, 0) for d in range(7)])
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Transactions')
    ax.set_title('Transaction Distribution by Day of Week')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig




