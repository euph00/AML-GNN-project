"""
Data loading utilities for IBM AML dataset.
Handles loading and preprocessing of transactions, accounts, and pattern data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

DEFAULT_DATASET_PATH = Path.home() / ".cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8"


def load_transactions(file_path: str = None, dataset_size: str = "HI-Small") -> pd.DataFrame:
    """
    Load transaction data.

    Args:
        file_path: Path to transactions CSV. If None, uses default kagglehub path
        dataset_size: One of "HI-Small", "HI-Medium", "HI-Large", "LI-Small", "LI-Medium", "LI-Large"

    Returns:
        DataFrame with columns:
        - timestamp: parsed datetime
        - from_bank, from_account: source bank ID and account ID
        - to_bank, to_account: target bank ID and account ID
        - amount_received, receiving_currency: received amount and currency
        - amount_paid, payment_currency: paid amount and currency
        - payment_format: type of payment
        - is_laundering: binary label
    """
    if file_path is None:
        file_path = DEFAULT_DATASET_PATH / f"{dataset_size}_Trans.csv"

    print(f"\nLoading transactions from: {file_path}")
    print(f"File size: {Path(file_path).stat().st_size / 1024**2:.1f} MB")

    df = pd.read_csv(
        file_path,
        parse_dates=['Timestamp'],
        dtype={
            'From Bank': str,
            'To Bank': str,
            'Account': str,
            'Payment Format': 'category',
            'Receiving Currency': 'category',
            'Payment Currency': 'category',
            'Is Laundering': np.int8
        }
    )

    # The data has duplicate account column, need rename to indicate direction of transaction
    df.columns = [
        'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
        'amount_received', 'receiving_currency', 'amount_paid',
        'payment_currency', 'payment_format', 'is_laundering'
    ]

    df.insert(0, 'transaction_id', range(len(df)))

    print(f"\nLoaded {len(df):,} transactions")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Laundering transactions: {df['is_laundering'].sum():,} ({df['is_laundering'].mean()*100:.3f}%)")

    return df


def load_accounts(file_path: str = None, dataset_size: str = "HI-Small") -> pd.DataFrame:
    """
    Load account metadata.

    Args:
        file_path: Path to accounts CSV. If None, uses default kagglehub path
        dataset_size: Dataset variant to load

    Returns:
        DataFrame with columns:
        - bank_name, bank_id: bank identifier
        - account_id: unique account identifier
        - entity_id, entity_name: owning entity information
    """
    if file_path is None:
        file_path = DEFAULT_DATASET_PATH / f"{dataset_size}_accounts.csv"

    print(f"\nLoading accounts from: {file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            'Bank Name': str,
            'Bank ID': str,
            'Account Number': str,
            'Entity ID': str,
            'Entity Name': str
        }
    )

    df.columns = ['bank_name', 'bank_id', 'account_id', 'entity_id', 'entity_name']

    print(f"\nLoaded {len(df):,} accounts from {df['bank_id'].nunique()} banks")

    return df


def load_patterns(file_path: str = None, dataset_size: str = "HI-Small") -> pd.DataFrame:
    """
    Load labeled laundering pattern data.

    Args:
        file_path: Path to patterns TXT file
        dataset_size: Dataset variant to load

    Returns:
        DataFrame with pattern information including:
        - pattern_id: sequential pattern identifier
        - pattern_type: type of laundering pattern
        - transaction data for each transaction in the pattern
    """
    if file_path is None:
        file_path = DEFAULT_DATASET_PATH / f"{dataset_size}_Patterns.txt"

    print(f"\nLoading patterns from: {file_path}")

    patterns = []
    current_pattern_id = 0
    current_pattern_type = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                parts = line.split(" - ")
                if len(parts) >= 2:
                    pattern_type_raw = parts[1].split(":")[0].strip()
                    current_pattern_type = pattern_type_raw
                    current_pattern_id += 1
            elif line.startswith("END LAUNDERING ATTEMPT"):
                continue
            else:
                parts = line.split(',')
                if len(parts) >= 10:
                    try:
                        patterns.append({
                            'pattern_id': current_pattern_id,
                            'pattern_type': current_pattern_type,
                            'timestamp': parts[0],
                            'from_bank': parts[1],
                            'from_account': parts[2],
                            'to_bank': parts[3],
                            'to_account': parts[4],
                            'amount_received': float(parts[5]),
                            'receiving_currency': parts[6],
                            'amount_paid': float(parts[7]),
                            'payment_currency': parts[8],
                            'payment_format': parts[9],
                            'is_laundering': int(parts[10]) if len(parts) > 10 else 1
                        })
                    except (ValueError, IndexError) as e:
                        continue

    df = pd.DataFrame(patterns)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print('\nLoaded patterns')

    return df


def get_dataset_summary(transactions_df: pd.DataFrame) -> Dict:
    """
    Generate high-level dataset statistics.

    Args:
        transactions_df: Loaded transactions DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_transactions': len(transactions_df),
        'laundering_transactions': transactions_df['is_laundering'].sum(),
        'laundering_ratio': transactions_df['is_laundering'].mean(),
        'date_range': {
            'start': transactions_df['timestamp'].min(),
            'end': transactions_df['timestamp'].max(),
            'days': (transactions_df['timestamp'].max() - transactions_df['timestamp'].min()).days
        },
        'unique_accounts': pd.concat([
            transactions_df['from_account'],
            transactions_df['to_account']
        ]).nunique(),
        'unique_banks': pd.concat([
            transactions_df['from_bank'],
            transactions_df['to_bank']
        ]).nunique(),
        'total_volume': {
            'received': transactions_df['amount_received'].sum(),
            'paid': transactions_df['amount_paid'].sum()
        },
        'currencies': transactions_df['receiving_currency'].unique().tolist(),
        'payment_formats': transactions_df['payment_format'].unique().tolist()
    }

    return summary
