import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configure the logging module
logging.basicConfig(
    filename='data_analysis.log',  # Name of the log file
    filemode='a',  # Append mode, which adds new log entries instead of overwriting
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO  # Logging level, e.g., INFO, DEBUG, ERROR
)

def plot_sales(data, x_col, y_col, estimator=np.sum, title='', x_label='', y_label=''):
    """
    Plots sales data using a bar plot.

    This function creates a bar plot for the given sales data. It allows customization of the plot 
    through various parameters like the estimator function, plot title, and axis labels.

    Args:
        data (DataFrame): The DataFrame containing the sales data.
        x_col (str): The name of the column in `data` to be used as the x-axis.
        y_col (str): The name of the column in `data` to be used as the y-axis.
        estimator (function, optional): The aggregation function to apply to `y_col`. Default is numpy.sum.
        title (str, optional): The title of the plot. Default is an empty string.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.

    Returns:
        None: This function does not return anything but plots the specified bar plot.
    """
    try:
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x=x_col, y=y_col, data=data, estimator=estimator)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        for bar in barplot.patches:
            barplot.annotate(format(bar.get_height(), '.2f'), 
                             (bar.get_x() + bar.get_width() / 2, bar.get_height()/2), 
                             ha='center', va='bottom',
                             size=10, xytext=(0, 8),
                             textcoords='offset points')
        plt.show()
        logging.info(f"Plot '{title}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to create plot '{title}' : {e}")
        plt.close() # Close the figure to prevent empty figures from showing up



def load_data(file_path):
    """
    Function to load data from a file into a pandas DataFrame.
    Supports CSV, Excel, JSON, and Parquet formats.

    Args:
        file_path ( str ):  Path to the file to be loaded.

    Returns:
         pandas.DataFrame: A pandas DataFrame if successful, None otherwise.
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
            logging.warning(f"Unsupported file format for file: {file_path}")
            return None
    except FileNotFoundError:
        logging.error(f"The file was not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"No data: The file is empty: {file_path}")
        return None
    except ValueError as ve:
        logging.error(f"Value Error: {ve} when loading file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def highlight_nans(val):
    """
    Highlights the cell in red if NaN or null

    Args:
        val (various): The value in the cell to be checked.

    Returns:
        str: A string representing the CSS to apply to the cell.
    """
    if pd.isna(val) or pd.isnull(val):
        logging.info(f"NaN or null value found: {val}")
        return "background-color: red"
    return ""


def convert_data_types(df, date_cols=[], numeric_cols=[], categorical_cols=[]):
    """
    Convert the data types of specified columns in a DataFrame.

    This function converts columns in a DataFrame to specified data types: datetime for date columns, 
    numeric for numerical columns, and category for categorical columns. If errors occur during conversion,
    they are coerced to NaN for numeric columns and NaT for datetime columns.

    Args:
        df (DataFrame): The pandas DataFrame whose columns are to be converted.
        date_cols (list of str, optional): List of column names to convert to datetime. Defaults to [].
        numeric_cols (list of str, optional): List of column names to convert to numeric types. Defaults to [].
        categorical_cols (list of str, optional): List of column names to convert to categorical. Defaults to [].

    Returns:
        DataFrame: The DataFrame with converted data types.

    Raises:
        Exception: If an error occurs during the conversion process.
    """
    try:
        logging.info("Starting data type conversion.")
        for col in date_cols:
            logging.info(f"Converting {col} to datetime.")
            df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in numeric_cols:
            logging.info(f"Converting {col} to numeric.")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in categorical_cols:
            logging.info(f"Converting {col} to category.")
            df[col] = df[col].astype("category")
        logging.info("Data type conversion successful.")
        return df
    except Exception as e:
        logging.error(f"Data type conversion error: {e}")
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
        logging.error(f"Column '{col}' not found in DataFrame.")
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    # For categorical data
    if valid_values is not None:
        if not isinstance(valid_values, list):
            logging.error("valid_values should be a list for categorical data.")
            raise TypeError("valid_values should be a list.")
        filtered_df = filter_categorical(df, col, valid_values)
        logging.info(f"Categorical filtering applied on '{col}'. Rows before: {len(df)}, Rows after: {len(filtered_df)}")
        return filtered_df

    # For numerical data
    if pd.api.types.is_numeric_dtype(df[col].dtype):
        filtered_df = filter_numerical(df, col, min_value, max_value)
        logging.info(f"Numerical filtering applied on '{col}'. Rows before: {len(df)}, Rows after: {len(filtered_df)}")
        return filtered_df

    logging.warning(f"No filtering applied on column: {col}")
    return df


def check_missing_values(df):
    """Return a DataFrame with the count of missing values in each column.

    Args:
        df (pandas.DataFrame): The DataFrame for which to count missing values.

    Returns:
        pandas.Series: A Series with the count of missing values for each column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    return df.isnull().sum()


def describe_data(df, cols):
    """
    Return descriptive statistics for specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        cols (list of str): The columns for which to provide descriptive statistics.

    Returns:
        pandas.DataFrame: A DataFrame containing descriptive statistics for the specified columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if not all(col in df.columns for col in cols):
        missing_cols = [col for col in cols if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
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
    if column not in df.columns:
        logging.error(f"Column '{column}' not found in DataFrame.")
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        logging.error(f"Column '{column}' is not numeric and cannot be processed for outliers.")
        raise TypeError(f"Column '{column}' must be numeric.")
    try:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        original_count = len(df)
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed_count = original_count - len(df_filtered)

        logging.info(f"Removed {removed_count} outliers from '{column}' using IQR method.")
        return df_filtered
    except Exception as e:
        logging.error(f"Error in handling outliers for '{column}': {e}")
        raise


def describe_statistics(df, cols=None):
    """
    Prints initial descriptive statistics of the DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to describe.
        cols (list, optional): List of columns to describe. If None, all columns are described.
    """
    if cols and not all(col in df.columns for col in cols):
        logging.warning("Some columns not found in the DataFrame.")
        return

    try:
        description = df.describe(include='all' if cols is None else cols)
        print("Initial Descriptive Statistics:")
        print(description)
        logging.info("Descriptive statistics successfully printed.")
    except Exception as e:
        logging.error(f"Error in describing data: {e}")

def analyze_categorical_columns(df, categorical_columns):
    """
    Prints the value counts of categorical columns.
    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        categorical_columns (list of str): List of categorical column names to analyze.
    """
    print("\nCategorical Columns Analysis:")
    for col in categorical_columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            logging.warning(f"Column '{col}' not found in DataFrame.")
            continue

        print(f"\nColumn: {col}")
        print(df[col].value_counts())
        logging.info(f"Value counts for column '{col}' printed.")


def find_rows_with_missing_values(df):
    """Checks and prints the count of missing values in each column

    Args:
        df (pandas.DataFrame): The DataFrame to check for missing values.

    Returns:
        str or pandas Styler: Message indicating no missing values or a DataFrame Styler highlighting rows with missing values.
    """
    rows_with_missing_values = df[df.isnull().any(axis=1)]
    if rows_with_missing_values.empty:
        logging.info(f"No missing values found in the DataFrame : {df}.")
        return f"No missing values exist in columns of {df}"
    
    logging.info(f"Found rows with missing values: {len(rows_with_missing_values)} rows")
    if len(rows_with_missing_values) < 30:
        return rows_with_missing_values.style.map(highlight_nans)
    else:
        logging.info(f"More than 30 rows with missing values found in: '{df}'.Not displaying dut to size")
        return "Rows with missing values exceed 30. Not displaying due to size."


def analyze_outliers(df, numeric_columns):
    """
    Analyzes and logs potential outliers in numeric columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        numeric_columns (list of str): List of numeric column names to analyze for outliers.
    """
    logging.info(f"Starting potential outliers analysis.")
    print("\n--- Potential Outliers Analysis ---")
    for col in numeric_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in DataFrame.")
            print(f"Column '{col}' not found in DataFrame.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            logging.warning(f"Column '{col}' is not numeric and cannot be analyzed for outliers.")
            print(f"Column '{col}' is not numeric and cannot be analyzed for outliers.")
            continue

        if df[col].max() / df[col].std() > 3:
            logging.warning(f"Outliers detected in column '{col}'.")
            print(
                f"The '{col}' column may contain outliers as indicated by a high max/standard deviation ratio."
            )
    logging.info("Outliers analysis completed.")

def kmeans_clustering(data, features, k_range):
    """
    Performs K-Means clustering for a given k range and returns the best k based on silhouette score.

    Args:
        data (pd.DataFrame): Your sales data.
        features (list): List of features to use for clustering.
        k_range (list): Range of k values to try.

    Returns:
        tuple: (kmeans_model, best_k, silhouette_score)
    """
    best_score = -1
    best_k = None
    best_model = None
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[features])
        score = silhouette_score(data[features], kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
    return best_model, best_k, best_score
