import numpy as np
import pandas as pd

def convert_currency_to_USD(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all transaction amounts to USD using exchange rates from September 1, 2022.

    Args:
        trans_df: DataFrame with columns amount_received, receiving_currency,
                  amount_paid, payment_currency

    Returns:
        DataFrame with amount_received and amount_paid converted to USD
    """

    # Exchange rates extracted from the dataset itself (1 unit of foreign currency = X USD)
    # These rates are consistent throughout the dataset and provide exact conversions
    exchange_rates_to_usd = {
        'Australian Dollar': 0.7078142565250762,
        'Bitcoin': 11882.606354334948,
        'Brazil Real': 0.17710085892926486,
        'Canadian Dollar': 0.7579777123340703,
        'Euro': 1.1717834945246437,
        'Mexican Peso': 0.04729675409908812,
        'Ruble': 0.012852809618619482,
        'Rupee': 0.013615816158909508,
        'Saudi Riyal': 0.26658846647162965,
        'Shekel': 0.29612079931906443,
        'Swiss Franc': 1.0928961748181611,
        'UK Pound': 1.2916559063890118,
        'US Dollar': 1.0,
        'Yen': 0.009487666034034777,
        'Yuan': 0.14930721433967664,
    }

    # Create a copy to avoid modifying the original
    df = trans_df.copy()

    # Convert amount_received to USD
    df['amount_received'] = df.apply(
        lambda row: row['amount_received'] * exchange_rates_to_usd.get(row['receiving_currency'], 1.0),
        axis=1
    )

    # Convert amount_paid to USD
    df['amount_paid'] = df.apply(
        lambda row: row['amount_paid'] * exchange_rates_to_usd.get(row['payment_currency'], 1.0),
        axis=1
    )

    return df


def normalize_amounts(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log scaling, zero centering, and normalization to transaction amounts.
    Uses the formula: amount_normalized = (log(amount + 1) - mean) / std

    Args:
        trans_df: DataFrame with columns amount_received and amount_paid (in USD)

    Returns:
        DataFrame with normalized amount_received and amount_paid columns
    """
    # Create a copy to avoid modifying the original
    df = trans_df.copy()

    # Apply log transform with +1 to handle zero values
    log_received = np.log(df['amount_received'] + 1)
    log_paid = np.log(df['amount_paid'] + 1)

    # Calculate mean and std for zero centering and normalization
    mean_received = log_received.mean()
    std_received = log_received.std()

    mean_paid = log_paid.mean()
    std_paid = log_paid.std()

    # Apply normalization: (log(amount + 1) - mean) / std
    df['amount_received'] = (log_received - mean_received) / std_received
    df['amount_paid'] = (log_paid - mean_paid) / std_paid

    return df


def temporal_encoding(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal encodings for the transaction timestamp:
    - Cyclical encoding for hour of day (sine and cosine)
    - Normalized linear time progression (unix timestamp standardized)

    Args:
        trans_df: DataFrame with 'timestamp' column (datetime64[ns])

    Returns:
        DataFrame with added columns:
        - hour_sin: sine encoding of hour (0-23)
        - hour_cos: cosine encoding of hour (0-23)
        - time_normalized: standardized unix timestamp
    """
    # Create a copy to avoid modifying the original
    df = trans_df.copy()

    # Extract hour of day (0-23)
    hour = df['timestamp'].dt.hour

    # Cyclical encoding: map hour to radians (0-23 hours -> 0-2π radians)
    hour_radians = 2 * np.pi * hour / 24

    # Sine and cosine components for cyclical encoding
    df['hour_sin'] = np.sin(hour_radians)
    df['hour_cos'] = np.cos(hour_radians)

    # Convert timestamp to unix timestamp (seconds since epoch)
    unix_timestamp = df['timestamp'].astype(np.int64) / 10**9  # Convert nanoseconds to seconds

    # Normalize unix timestamp: (timestamp - mean) / std
    mean_time = unix_timestamp.mean()
    std_time = unix_timestamp.std()
    df['time_normalized'] = (unix_timestamp - mean_time) / std_time

    return df

def encode_currency_ids(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode currency categories as numerical IDs.

    Args:
        trans_df: DataFrame with 'payment_currency' and 'receiving_currency' columns

    Returns:
        DataFrame with added columns:
        - payment_currency_id: numerical ID (0 to n-1) for payment currency
        - receiving_currency_id: numerical ID (0 to n-1) for receiving currency
    """
    # Create a copy to avoid modifying the original
    df = trans_df.copy()

    # Get all unique currencies from both columns
    all_currencies = sorted(set(df['payment_currency'].unique()) | set(df['receiving_currency'].unique()))

    # Create mapping from currency name to ID
    currency_to_id = {currency: idx for idx, currency in enumerate(all_currencies)}

    # Map currencies to their IDs
    df['payment_currency_id'] = df['payment_currency'].map(currency_to_id)
    df['receiving_currency_id'] = df['receiving_currency'].map(currency_to_id)

    return df


def encode_payment_format_ids(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode payment format categories as numerical IDs.

    Args:
        trans_df: DataFrame with 'payment_format' column

    Returns:
        DataFrame with added column:
        - payment_format_id: numerical ID (0 to n-1) for payment format
    """
    # Create a copy to avoid modifying the original
    df = trans_df.copy()

    # Get all unique payment formats and sort them
    all_formats = sorted(df['payment_format'].unique())

    # Create mapping from payment format to ID
    format_to_id = {format_name: idx for idx, format_name in enumerate(all_formats)}

    # Map payment formats to their IDs
    df['payment_format_id'] = df['payment_format'].map(format_to_id)

    return df


def encode_account_ids(trans_df: pd.DataFrame) -> tuple:
    """
    Encode account identifiers as numerical IDs.

    Args:
        trans_df: DataFrame with 'from_account' and 'to_account' columns

    Returns:
        Tuple containing:
        - DataFrame with added columns:
            - from_account_id: numerical ID (0 to n-1) for source account
            - to_account_id: numerical ID (0 to n-1) for destination account
        - account_to_id: dict mapping account hex string to numerical ID
        - id_to_account: dict mapping numerical ID to account hex string
    """
    # Create a copy to avoid modifying the original
    df = trans_df.copy()

    # Get all unique accounts from both columns
    all_accounts = sorted(set(df['from_account'].unique()) | set(df['to_account'].unique()))

    # Create bidirectional mappings
    account_to_id = {account: idx for idx, account in enumerate(all_accounts)}
    id_to_account = {idx: account for idx, account in enumerate(all_accounts)}

    # Map accounts to their IDs
    df['from_account_id'] = df['from_account'].map(account_to_id)
    df['to_account_id'] = df['to_account'].map(account_to_id)

    return df, account_to_id, id_to_account


def decode_account_ids(account_ids: np.ndarray, id_to_account: dict) -> list:
    """
    Decode numerical account IDs back to their hexadecimal string names.

    Args:
        account_ids: Array or list of numerical account IDs
        id_to_account: Dictionary mapping numerical ID to account hex string

    Returns:
        List of account hex strings corresponding to the input IDs
    """
    return [id_to_account[account_id] for account_id in account_ids]


def temporal_train_test_split(trans_df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Split transactions into train and test sets based on temporal ordering.
    The first train_ratio of transactions (by time) go to train, rest to test.

    Args:
        trans_df: DataFrame with 'timestamp' and 'is_laundering' columns
        train_ratio: Fraction of transactions for training set (default 0.8)

    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by timestamp to maintain temporal ordering
    df_sorted = trans_df.sort_values('timestamp').reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df_sorted) * train_ratio)

    # Split into train and test
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Print split statistics
    print("=" * 60)
    print("TEMPORAL TRAIN/TEST SPLIT")
    print("=" * 60)
    print(f"\nTrain Set:")
    print(f"  Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"  Transactions: {len(train_df):,}")
    print(f"  Laundering: {train_df['is_laundering'].sum():,} ({train_df['is_laundering'].mean()*100:.3f}%)")

    print(f"\nTest Set:")
    print(f"  Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"  Transactions: {len(test_df):,}")
    print(f"  Laundering: {test_df['is_laundering'].sum():,} ({test_df['is_laundering'].mean()*100:.3f}%)")
    print("=" * 60)

    return train_df, test_df


def compute_account_statistics(trans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute account-level statistics for each transaction.
    Adds 16 features: 7 for source account, 7 for destination account, 2 transaction-relative.

    Note: Expects amount_paid and amount_received in USD (not yet normalized).

    Args:
        trans_df: DataFrame with transaction data including account IDs, timestamps, and USD amounts

    Returns:
        DataFrame with 16 additional normalized account statistics columns:
        - from_tx_rate, to_tx_rate: transactions per day (log + z-score normalized)
        - from_avg_amount, to_avg_amount: mean amount (log + z-score normalized)
        - from_std_amount, to_std_amount: std dev of amounts (log + z-score normalized)
        - from_unique_destinations, to_unique_sources: unique partners (log + z-score normalized)
        - from_destination_diversity, to_source_diversity: diversity ratios [0,1]
        - from_days_active, to_days_active: days between first/last tx (z-score normalized)
        - from_is_first_tx, to_is_first_tx: boolean flags {0,1}
        - amount_zscore_from, amount_zscore_to: transaction amount z-scores
    """
    df = trans_df.copy()

    # Compute source account (from_account_id) statistics
    from_stats = df.groupby('from_account_id').agg({
        'transaction_id': 'count',
        'to_account_id': 'nunique',
        'amount_paid': ['mean', 'std'],
        'timestamp': ['min', 'max']
    })

    from_stats.columns = ['_from_total_tx', '_from_unique_destinations',
                          '_from_avg_amount', '_from_std_amount',
                          '_from_min_time', '_from_max_time']

    from_stats['_from_days_active'] = (from_stats['_from_max_time'] - from_stats['_from_min_time']).dt.total_seconds() / 86400 + 1
    from_stats['_from_tx_rate'] = from_stats['_from_total_tx'] / from_stats['_from_days_active']
    from_stats['from_destination_diversity'] = from_stats['_from_unique_destinations'] / from_stats['_from_total_tx']

    # Compute destination account (to_account_id) statistics
    to_stats = df.groupby('to_account_id').agg({
        'transaction_id': 'count',
        'from_account_id': 'nunique',
        'amount_received': ['mean', 'std'],
        'timestamp': ['min', 'max']
    })

    to_stats.columns = ['_to_total_tx', '_to_unique_sources',
                        '_to_avg_amount', '_to_std_amount',
                        '_to_min_time', '_to_max_time']

    to_stats['_to_days_active'] = (to_stats['_to_max_time'] - to_stats['_to_min_time']).dt.total_seconds() / 86400 + 1
    to_stats['_to_tx_rate'] = to_stats['_to_total_tx'] / to_stats['_to_days_active']
    to_stats['to_source_diversity'] = to_stats['_to_unique_sources'] / to_stats['_to_total_tx']

    # Merge statistics back to transactions
    df = df.merge(from_stats, left_on='from_account_id', right_index=True, how='left')
    df = df.merge(to_stats, left_on='to_account_id', right_index=True, how='left')

    # Identify first transactions (binary flags)
    df['from_is_first_tx'] = (df['timestamp'] == df['_from_min_time']).astype(int)
    df['to_is_first_tx'] = (df['timestamp'] == df['_to_min_time']).astype(int)

    # Compute transaction-relative z-scores
    df['amount_zscore_from'] = np.where(
        (df['_from_std_amount'] > 0) & (~df['_from_std_amount'].isna()),
        (df['amount_paid'] - df['_from_avg_amount']) / df['_from_std_amount'],
        0
    )

    df['amount_zscore_to'] = np.where(
        (df['_to_std_amount'] > 0) & (~df['_to_std_amount'].isna()),
        (df['amount_received'] - df['_to_avg_amount']) / df['_to_std_amount'],
        0
    )

    # Apply log + z-score normalization to right-skewed features
    for col in ['_from_tx_rate', '_to_tx_rate',
                '_from_avg_amount', '_to_avg_amount',
                '_from_std_amount', '_to_std_amount',
                '_from_unique_destinations', '_to_unique_sources']:
        log_values = np.log(df[col] + 1)
        mean_val = log_values.mean()
        std_val = log_values.std()
        normalized_name = col[1:]  # Remove leading underscore
        df[normalized_name] = (log_values - mean_val) / std_val

    # Apply z-score normalization to days active
    for col in ['_from_days_active', '_to_days_active']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        normalized_name = col[1:]  # Remove leading underscore
        df[normalized_name] = (df[col] - mean_val) / std_val

    # Drop temporary columns
    temp_cols = [col for col in df.columns if col.startswith('_')]
    df = df.drop(columns=temp_cols)

    # Fill NaN values with 0 (occurs for single-transaction accounts where std is undefined)
    account_stat_cols = [
        'from_tx_rate', 'from_avg_amount', 'from_std_amount', 'from_unique_destinations',
        'from_destination_diversity', 'from_days_active',
        'to_tx_rate', 'to_avg_amount', 'to_std_amount', 'to_unique_sources',
        'to_source_diversity', 'to_days_active',
        'amount_zscore_from', 'amount_zscore_to'
    ]
    df[account_stat_cols] = df[account_stat_cols].fillna(0)

    return df
