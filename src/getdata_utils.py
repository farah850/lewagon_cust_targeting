import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath="../data/bank-full.csv"):
    """
    Loads the bank dataset, processes the target variable, and splits the data into
    training, validation, and test sets.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_val (pd.Series): Validation labels
        y_test (pd.Series): Test labels
    """
    # Step 1: Load the data
    df_bank = pd.read_csv(filepath, sep=';', header=0)

    # Step 2: Prepare features and target
    X = df_bank.drop(['y', 'duration'], axis=1)
    y = df_bank['y'].map({'no': 0, 'yes': 1})

    # Step 3: Split into temp (train+val) and test (80/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Step 4: Split temp into train and val (75/25 of 80% = 60/20/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
