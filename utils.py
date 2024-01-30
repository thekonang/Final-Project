import pandas as pd


def load_data(file_path):
    """
    Function to load data from a file into a pandas DataFrame.
    Supports CSV, Excel, JSON, and Parquet formats.

    Args:
        file_path ( str ): _description_

    Returns:
        _type_: _description_
    """
    try:
        if file_path.endswith(".csv"):  # Load a CSV file
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)  # Load an Excel fileσ
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)  # Load a JSON file
        elif file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)  # Load a Parquet file
        else:
            print(
                f"File format not supported: {file_path}"
            )  # File format not supported

            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def highlight_nans(val):
    """
    Highlights the cell in red if NaN or null

    Args:
        val (_type_): _description_

    Returns:
        _type_: _description_
    """
    color = "red" if pd.isna(val) or pd.isnull(val) else "None"
    return f"background-color: {color}"


def convert_data_types(df, date_cols=[], numeric_cols=[], categorical_cols=[]):
    """Convert columns to appropriate data types.

    Args:
        df (_type_): _description_
        date_cols (list, optional): _description_. Defaults to [].
        numeric_cols (list, optional): _description_. Defaults to [].
        categorical_cols (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    return df


def filter_data(df, col, valid_values=None, min_value=None, max_value=None):
    """
    Filters a DataFrame based on the valid values for a given column.

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        col (str): The name of the column to filter on.
        valid_values (list, optional): A list of valid values for categorical data.
        min_value (int, optional): The minimum value for numerical data.
        max_value (int, optional): The maximum value for numerical data.

    Returns:
        pandas.DataFrame: A DataFrame filtered based on the specified criteria.
    """

    def filter_categorical(dataframe, column, values):
        filtered_df = dataframe[dataframe[column].isin(values)].copy()  # Explicit copy
        filtered_df[column] = filtered_df[column].cat.remove_unused_categories()
        return filtered_df

    def filter_numerical(dataframe, column, min_val, max_val):
        if min_val is not None and max_val is not None:
            return dataframe[(dataframe[column] >= min_val) & (dataframe[column] <= max_val)]
        if min_val is not None :
            return dataframe[dataframe[column] >= min_val]
        if max_val is not None:
            return dataframe[dataframe[column] <= max_val]
        return dataframe

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    data_type_handlers = {
        'object': lambda df, col, val: filter_categorical(df, col, val),
        'category': lambda df, col, val: filter_categorical(df, col, val),
        'numeric': lambda df, col, min_val, max_val: filter_numerical(df, col, min_val, max_val)
    }

    data_type = df[col].dtype
    if data_type == 'object' or data_type.name == 'category':
        return data_type_handlers[data_type.name](df, col, valid_values)
    elif pd.api.types.is_numeric_dtype(data_type):
        return data_type_handlers['numeric'](df, col, min_value, max_value)

    return df


def check_missing_values(df):
    """Return a DataFrame with the count of missing values in each column.

    Args:
        df (pandas DataFrame): _description_

    Returns:
        _type_: _description_
    """
    return df.isnull().sum()


def describe_data(df, cols):
    """Return descriptive statistics for specified columns
    Args:
        df (DataFrame): The DataFrame containing the data.
        cols ( array ): _description_

    Returns:
        _type_: _description_
    """
    return df[cols].describe()


def handle_outliers_IQR(df, column, factor=1.5):
    """
    Handles outliers in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    Args:
        df (pandas DataFrame): The DataFrame containing the data.
        column (str): The name of the column to check for outliers.
        factor (float, optional): The factor to multiply with IQR for determining the bounds. Defaults to 1.5.

    Returns:
        pandas DataFrame: The DataFrame with outliers handled.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df


def describe_statistics(df):
    """Prints initial descriptive statistics of the DataFrame

    Args:
        df (pandas DatFrame): _description_
    """
    print("Initial Descriptive Statistics:")
    print(df.describe(include="all"))


def analyze_categorical_columns(df, categorical_columns):
    """Prints the value counts of categorical columns

    Args:
        df (_type_): _description_
        categorical_columns (_type_): _description_
    """
    print("\nCategorical Columns Analysis:")
    for col in categorical_columns:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())


def find_rows_with_missing_values(df):
    """Checks and prints the count of missing values in each column

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    rows_with_missing_values = df[df.isnull().any(axis=1)]
    if rows_with_missing_values.empty:
        return f"No missing values exist in columns of {df}"
    elif (
        len(rows_with_missing_values) < 30
    ):  # Assuming The length of 30 rows is viable to be checked on the spot
        # Return columns with missing values
        return rows_with_missing_values.style.applymap(highlight_nans)


def analyze_outliers(df, numeric_columns):
    """Analyzes and prints potential outliers in numeric columns

    Args:
        df (_type_): _description_
        numeric_columns (_type_): _description_
    """
    print("\n--- Potential Outliers Analysis ---")
    for col in numeric_columns:
        if df[col].max() / df[col].std() > 3:
            print(
                f"The '{col}' column may contain outliers as indicated by a high max/standard deviation ratio."
            )
