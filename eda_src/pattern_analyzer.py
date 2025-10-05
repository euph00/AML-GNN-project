"""
Pattern detection utilities for identifying money laundering structures.
"""

import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict


def analyze_pattern_characteristics(
    pattern_df: pd.DataFrame,
    transactions_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute summary statistics for each laundering pattern.

    Args:
        pattern_df: DataFrame with pattern information
        transactions_df: Full transaction data (optional)

    Returns:
        DataFrame with pattern statistics
    """
    print("Analyzing pattern characteristics...")

    pattern_stats = []

    for pattern_id in pattern_df['pattern_id'].unique():
        pattern_trans = pattern_df[pattern_df['pattern_id'] == pattern_id]

        unique_accounts = set()
        unique_accounts.update(pattern_trans['from_account'].values)
        unique_accounts.update(pattern_trans['to_account'].values)

        stats = {
            'pattern_id': pattern_id,
            'pattern_type': pattern_trans['pattern_type'].iloc[0],
            'num_accounts': len(unique_accounts),
            'num_transactions': len(pattern_trans),
            'total_amount': pattern_trans['amount_received'].sum(),
            'avg_amount': pattern_trans['amount_received'].mean(),
            'time_span_hours': (pattern_trans['timestamp'].max() - pattern_trans['timestamp'].min()).total_seconds() / 3600
        }
        pattern_stats.append(stats)

    pattern_stats = pd.DataFrame(pattern_stats)

    print(f"  Analyzed {len(pattern_stats)} patterns")
    print(f"\nPattern type distribution:")
    print(pattern_stats['pattern_type'].value_counts())

    return pattern_stats


def get_pattern_examples(
    pattern_df: pd.DataFrame,
    pattern_type: str,
    num_examples: int = 3
) -> List[int]:
    """
    Retrieve sample instances of a pattern type.

    Args:
        pattern_df: DataFrame with pattern information
        pattern_type: Type of pattern to retrieve
        num_examples: Number of examples to return

    Returns:
        List of pattern_ids
    """
    matching_patterns = pattern_df[pattern_df['pattern_type'] == pattern_type]

    if len(matching_patterns) == 0:
        print(f"No patterns found for type: {pattern_type}")
        return []

    pattern_ids = matching_patterns['pattern_id'].unique()

    return pattern_ids[:num_examples].tolist()


def get_pattern_accounts(pattern_df: pd.DataFrame, pattern_id: int) -> List[str]:
    """
    Get all accounts involved in a specific pattern.

    Args:
        pattern_df: DataFrame with pattern information
        pattern_id: Pattern identifier

    Returns:
        List of unique account IDs
    """
    pattern_trans = pattern_df[pattern_df['pattern_id'] == pattern_id]

    accounts = set()
    accounts.update(pattern_trans['from_account'].values)
    accounts.update(pattern_trans['to_account'].values)

    return list(accounts)

"""
Technical consideration: how to get the best node for visualization using ego network?

1. When setting up examples for section 6.3, we run into issues for visualization, for example when we select some arbitrary node in a FAN-OUT pattern, we could select a peripheral node. In that case, no matter how high a radius we set for the ego network, we end up with just one red node and no edges coming out of it. This is not meaningful. 

2. In fact it is more common to run into cases of this problem, as only convenient patterns like CYCLE are initialization agnostic. Thus we need some clever way to initialize which node to use as the starting node to do visualization.
"""
def get_pattern_center_node(pattern_df: pd.DataFrame, pattern_id: int) -> str:
    """
    Select the appropriate node to generate an ego network for its laundering pattern.

    Args:
        pattern_df: DataFrame with pattern information
        pattern_id: Pattern identifier

    Returns:
        Account ID of the best(?) center node for visualization
    """
    pattern_trans = pattern_df[pattern_df['pattern_id'] == pattern_id]

    if len(pattern_trans) == 0:
        raise ValueError(f"No transactions found for pattern {pattern_id}")

    pattern_type = pattern_trans['pattern_type'].iloc[0]

    from_counts = pattern_trans['from_account'].value_counts()
    to_counts = pattern_trans['to_account'].value_counts()

    all_accounts = set(pattern_trans['from_account'].tolist() +
                       pattern_trans['to_account'].tolist())

    total_degree = {}
    for acc in all_accounts:
        out_deg = from_counts.get(acc, 0)
        in_deg = to_counts.get(acc, 0)
        total_degree[acc] = out_deg + in_deg

    if pattern_type == 'FAN-OUT':
        #max outdegree is correct as that is the center of the star
        center = from_counts.idxmax()

    elif pattern_type == 'FAN-IN':
        # max indegree is correct as that is the center of the star
        center = to_counts.idxmax()

    elif pattern_type == 'SCATTER-GATHER':
        # same as FAN-OUT and FAN-IN
        center = from_counts.idxmax()

    elif pattern_type == 'GATHER-SCATTER':
        # this is actually not really correct, as there is not one node in particular that can represent a good starting point. In practice, this results in the GATHER-SCATTER pattern looking like a FAN-OUT, as we fail to capture the GATHER edges in this ego network
        center = to_counts.idxmax()

    elif pattern_type == 'CYCLE':
        #any
        center = pattern_trans.iloc[0]['from_account']

    else:
        # Choose node with highest total degree
        center = max(total_degree, key=total_degree.get)

    return center

"""
Technical considerations: correct hop length for visualization?

1. Different patterns have different requirements for ego network radius. With too high a value we get a lot of noise (non-laundering edges) and with too low a value the pattern is not complete. This is important because it will affect how we do graph construction for subgraphs when training the GNN as well.

2. FAN-OUT/IN only really require 1 hop

3. CYCLE requires as many hops as there are edges in the cycle, which is quite a large range of 2 to 12(refer to summary stats for the patterns in the notebook)

4. GATHER-SCATTER and vice versa are more or less same as FAN-OUT/IN

5. BIPARTITIE only requires 1 hop with more careful selection of starting nodes needed

6. RANDOM ????? good luck bro
"""
def get_pattern_visualization_radius(pattern_df: pd.DataFrame, pattern_id: int) -> int:
    """
    Determine appropriate ego network radius for visualizing a pattern.

    Args:
        pattern_df: DataFrame with pattern information
        pattern_id: Pattern identifier

    Returns:
        Integer radius value for ego network extraction
    """
    pattern_trans = pattern_df[pattern_df['pattern_id'] == pattern_id]

    if len(pattern_trans) == 0:
        raise ValueError(f"No transactions found for pattern {pattern_id}")

    pattern_type = pattern_trans['pattern_type'].iloc[0]

    RADIUS_MAP = {
        'FAN-OUT': 2,
        'FAN-IN': 2,
        'CYCLE': 10,
        'GATHER-SCATTER': 2,
        'SCATTER-GATHER': 3,
        'STACK': 2,
        'BIPARTITE': 2,
        'RANDOM': 6
    }

    radius = RADIUS_MAP.get(pattern_type, 3) #3 for default

    return radius
