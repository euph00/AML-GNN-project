"""
Feature engineering utilities for transaction data analysis.
"""

import pandas as pd



def compute_account_statistics(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-account metrics.

    Returns:
        DataFrame with account-level statistics
    """
    print("Computing account statistics...")

    #aggregate outgoing transactions
    outgoing = transactions_df.groupby('from_account').agg({
        'transaction_id': 'count',
        'amount_paid': ['sum', 'mean', 'std', 'min', 'max'],
        'to_account': 'nunique',
        'is_laundering': 'sum'
    })
    outgoing.columns = ['num_outgoing', 'total_sent', 'avg_sent', 'std_sent', 'min_sent', 'max_sent', 'num_recipients', 'laundering_sent']

    #aggregate incoming transactions
    incoming = transactions_df.groupby('to_account').agg({
        'transaction_id': 'count',
        'amount_received': ['sum', 'mean', 'std', 'min', 'max'],
        'from_account': 'nunique',
        'is_laundering': 'sum'
    })
    incoming.columns = ['num_incoming', 'total_received', 'avg_received', 'std_received', 'min_received', 'max_received', 'num_senders', 'laundering_received']

    #combine
    account_stats = outgoing.join(incoming, how='outer').fillna(0)
    account_stats['total_transactions'] = account_stats['num_outgoing'] + account_stats['num_incoming']
    account_stats['net_flow'] = account_stats['total_received'] - account_stats['total_sent']

    print(f"  Computed statistics for {len(account_stats):,} accounts")

    return account_stats