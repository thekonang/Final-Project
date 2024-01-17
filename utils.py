import pandas as pd

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def convert_data_types(df, date_cols=[], numeric_cols=[], categorical_cols=[]):
    """Convert columns to appropriate data types."""
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df

def filter_data(df, col, valid_values):
    """Filter DataFrame based on valid values in a column."""
    return df[df[col].isin(valid_values)]

def check_missing_values(df):
    """Return a DataFrame with the count of missing values in each column."""
    return df.isnull().sum()

def describe_data(df, cols):
    """Return descriptive statistics for specified columns."""
    return df[cols].describe()

def handle_outliers_IQR(df, column,  factor=1.5):
    """
    Handles outliers in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The name of the column to check for outliers.
    method (str, optional): The method used for outlier detection, default is 'IQR'.
    factor (float, optional): The factor to multiply with IQR for determining the bounds, default is 1.5.

    Returns:
    DataFrame: The DataFrame with outliers handled.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

def describe_statistics(df):
    """Prints initial descriptive statistics of the DataFrame."""
    print("Initial Descriptive Statistics:")
    print(df.describe(include='all'))

def analyze_categorical_columns(df, categorical_columns):
    """Prints the value counts of categorical columns."""
    print("\nCategorical Columns Analysis:")
    for col in categorical_columns:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())

def check_and_print_missing_values(df):
    """Checks and prints the count of missing values in each column."""
    missing_values = df.isnull().sum()
    print("\nInitial Missing Values:")
    print(missing_values)
        # Return columns with missing values
    return missing_values[missing_values > 0].index.tolist()

def analyze_outliers(df, numeric_columns):
    """Analyzes and prints potential outliers in numeric columns."""
    print("\n--- Potential Outliers Analysis ---")
    for col in numeric_columns:
        if df[col].max() / df[col].std() > 3:
            print(f"The '{col}' column may contain outliers as indicated by a high max/standard deviation ratio.")