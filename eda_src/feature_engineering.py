"""
Feature engineering utilities for transaction data analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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

def apply_feature_eng_transactions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the transactions DataFrame.
    """
    # Normalize timestamps to start from zero
    transactions_df['timestamp'] = transactions_df['timestamp'] - transactions_df['timestamp'].min()

    # Convert zero-padded bank ids to ints
    transactions_df['from_bank'] = transactions_df['from_bank'].astype(int)
    transactions_df['to_bank'] = transactions_df['to_bank'].astype(int)

    # Create unique bank-account identifiers (cannot purely rely on from_account and to_account being unique only within a bank)
    transactions_df['from_bank_account_id'] = transactions_df['from_bank'].astype(str) + '_' + transactions_df['from_account']
    transactions_df['to_bank_account_id'] = transactions_df['to_bank'].astype(str) + '_' + transactions_df['to_account']

    # Add Amount features (log transform to reduce skew)
    transactions_df['log_amount_paid'] = np.log10(transactions_df['amount_paid'])
    transactions_df['log_amount_received'] = np.log10(transactions_df['amount_received'])

    # Add Amount ratio and difference
    transactions_df['amount_ratio'] = transactions_df['amount_paid'] / (transactions_df['amount_received'] + 1e-6)
    transactions_df['amount_diff'] = transactions_df['amount_paid'] - transactions_df['amount_received']

    columns_to_scale = ['amount_ratio', 'amount_diff']

    scaler = StandardScaler()
    scaler.fit(transactions_df[columns_to_scale])
    transactions_df[columns_to_scale] = scaler.transform(transactions_df[columns_to_scale])

    # Add Currency matching flags
    transactions_df['currency_match'] = (transactions_df['payment_currency'] == transactions_df['receiving_currency']).astype(int)

    # Add Bank relationship features
    transactions_df['same_bank'] = (transactions_df['from_bank'] == transactions_df['to_bank']).astype(int)

    # Select relevant columns for edge features
    transactions_df = transactions_df[['from_bank_account_id', 'to_bank_account_id',
                                       'log_amount_paid', 'log_amount_received',
                                       'amount_ratio', 'amount_diff',
                                       'currency_match', 'same_bank',
                                    #    'payment_format', 'receiving_currency', 'payment_currency',
                                       'is_laundering']]

    # One-hot encoding to payment_format, receiving_currency, payment_currency
    # transactions_df = pd.get_dummies(transactions_df, columns=['payment_format', 'receiving_currency', 'payment_currency'], dtype=int)

    return transactions_df

def apply_feature_eng_accounts(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the accounts DataFrame.
    """

    # Create unique bank-account identifier to link with transactions
    accounts_df['bank_account_id'] = accounts_df['bank_id'] + '_' + accounts_df['account_id']

    # Extract entity type from entity_name
    accounts_df[['entity_type', 'entity_number']] = accounts_df['entity_name'].str.split(' #', expand=True)

    # Keep only selected columns for node features
    accounts_df = accounts_df[['bank_account_id','entity_type']].drop_duplicates()

    # One-hot encoding to entity_type
    accounts_df = pd.get_dummies(accounts_df, columns=['entity_type'], dtype=int)

    return accounts_df