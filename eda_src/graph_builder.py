"""
Graph construction utilities for transaction network analysis.
Builds NetworkX graphs from transaction data.
"""

import pandas as pd
import networkx as nx
from typing import Dict
import numpy as np

"""
Technical Considerations: Graph construction

1. Graph type selection: Multi Directed Graph (MultiDiGraph)
For the AML usecase, there is a requirement for the graph to distinguish between in and outdegree. If the graph were undirected we would lose the ability to distinguish "A pays B" vs "B pays A". Further, this complicates pattern detection as FAN-OUT and FAN-IN both just appear as high degree, and CYCLE detection will find useless undirected loops that mean nothing.
Further, we require the use of a multigraph as we need multiple edges between the same pair of nodes, since the same accounts can transact multiple times. This allows us to model the temporal sequence of each transaction and preserve individual transaction attributes without any aggregation into a single edge that would cause information loss of possibly important features that differentiate legitimate and illegitimate transactions.
That said, there are usecases where we need different graph types. In networkx, PageRank requires a simple graph and Louvain requires undirected graphs. We might need these for secondary level analysis (Louvain: are laundering accounts clustered in communities? PageRank: Are important nodes more or less likely to be laundering accounts?). In these cases we can do things like G_simple = nx.DiGraph(G) and G_undirected = G.toundirected(), although that incurs a nontrivial amount of extra memory for larger graphs.

2. Graph design
We model accounts as nodes and transactions as edges. This is an obvious choice as laundering patterns are account-account interaction patterns. Thus, node features such as degree and centrality will directly represent account behavior and all other important information such as transaction characteristics are encoded in the edges.
As a side note, we do not immediately populate node features (total transactions, total sent, avg received etc) for each account immediately during graph construction. This is handled by feature_engineering.py, as most of this can only be efficiently calculated after the graph is constructed.

"""

def build_transaction_graph(
    transactions_df: pd.DataFrame,
) -> nx.MultiDiGraph:
    """
    Construct NetworkX graph from transactions.

    Args:
        transactions_df: DataFrame with transaction data
        directed: Whether to create directed edges (default True)
        multigraph: Whether to allow multiple edges between nodes (default True)

    Returns:
        NetworkX MultiDiGraph with transactions as edges
    """
    print(f"Building transaction graph...")

    G = nx.MultiDiGraph()

    for idx, row in transactions_df.iterrows():
        G.add_edge(
            row['from_account'],
            row['to_account'],
            transaction_id=row.get('transaction_id', idx),
            timestamp=row['timestamp'],
            from_bank=row['from_bank'],
            to_bank=row['to_bank'],
            amount=row['amount_received'],
            currency=row['receiving_currency'],
            payment_format=row['payment_format'],
            is_laundering=row['is_laundering'] # ground truth
        )

    print(f"  Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    return G


def extract_ego_network(
    graph: nx.MultiDiGraph,
    center_node: str,
    radius: int = 3
) -> nx.MultiDiGraph:
    """
    Extract k-hop neighborhood around a node.

    Args:
        center_node: Account ID to center on
        radius: Number of hops

    Returns:
        Ego network subgraph
    """
    if center_node not in graph:
        raise ValueError(f"Node {center_node} not in graph")

    # Get ego graph (NetworkX function)
    ego_graph = nx.ego_graph(graph, center_node, radius=radius)

    print(f"Ego network for {center_node} (radius={radius}): {ego_graph.number_of_nodes()} nodes, {ego_graph.number_of_edges()} edges")

    return ego_graph


"""
Technical Consideratios: graph statistics

1. As a starter, we calculate common graph stats that we would like to know such as density, degree summary stats and connected components. These allow us to profile subgraphs with reference to one another, which can possibly be used as discriminating features for suspicious networks.

This function is sufficiently modular imo. Please feel free to add on other things you may be interested in.
"""
def calculate_graph_statistics(graph: nx.MultiDiGraph) -> Dict:
    """
    Compute basic graph metrics.

    Args:
        graph: Transaction graph

    Returns:
        Dictionary with graph statistics
    """
    print("Calculating graph statistics...")

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Graph density calc
    if num_nodes > 1:
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0
    else:
        density = 0

    # Degree calc
    if graph.is_directed():
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        avg_in_degree = np.mean(in_degrees) if in_degrees else 0
        avg_out_degree = np.mean(out_degrees) if out_degrees else 0
        avg_degree = (avg_in_degree + avg_out_degree) / 2
    else:
        degrees = [d for n, d in graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0

    # Connected components calc
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    num_components = len(components)
    largest_component_size = len(max(components, key=len)) if components else 0

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'num_components': num_components,
        'largest_component_size': largest_component_size,
        'largest_component_fraction': largest_component_size / num_nodes if num_nodes > 0 else 0
    }

    if graph.is_directed():
        stats['avg_in_degree'] = avg_in_degree
        stats['avg_out_degree'] = avg_out_degree

    print("  Statistics calculated:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value:,}")

    return stats
