# Data Science Project: Retail Sales Analysis

## Overview

This project involves a comprehensive analysis of a fictional retail company's sales data. The goal is to apply data cleaning, descriptive statistics, inferential statistics, data visualization, and predictive modeling techniques to gain insights into sales trends, customer demographics, and product performance.

## Dataset Description

The dataset `adjusted_retail_sales_data_v2.csv` includes the following columns:

- `SalesDate`: Date of the sale (YYYY-MM-DD).
- `ProductCategory`: Category of the product (Clothing, Electronics, Home Appliances).
- `SalesAmount`: Sales amount in USD.
- `CustomerAge`: Age of the customer (Categorical).
- `CustomerGender`: Gender of the customer (Male, Female, Non-binary).
- `CustomerLocation`: Location of the customer (Japan, Australia, India, USA, UK, Canada).
- `ProductRatings`: Product rating (1 to 5).

# Prerequisites

- Python 3
- Libraries: pandas, numpy, matplotlib, seaborn (for data analysis and visualization)
- Jupyter Notebook or similar Python IDE

## Installation and Setup
To run this project, you will need to install its dependencies. This project uses a requirements.txt file to manage these dependencies.
Instructions for setting up the project environment:

1. Clone the Repository: First, clone this repository to your local machine using:
```bash
git clone https://github.com/thekonang/Final-Project-TechPro-DS.git
```

2. Install Dependencies: Navigate to the project directory and install the required Python packages:
```bash
cd Final-Project-TechPro-DS
pip install -r requirements.txt
```
This command will install all the packages listed in the requirements.txt file.

# Running the Project:
After installing the dependencies, you can run the project with: 
```bash
jupyter notebook DataAnalysis.ipynb
```
# Project Structure

- DataAnalysis.ipynb: Jupyter Notebook containing the data analysis, visualization, and modeling.
- data/: Directory containing the dataset.
- utils.py: Python file containing helper functions for data analysis tasks such as data loading, filtering, visualization, and outlier handling.


Data Cleaning and Preparation
The data cleaning process involves handling missing values, addressing outliers, correcting data types, and ensuring data consistency. Detailed steps are outlined in the data_cleaning.py script.

Analysis
The project includes descriptive statistics, data visualization, inferential statistics, predictive modeling, and advanced statistical analysis. Jupyter notebooks are used for this purpose.
escriptive statistics were calculated using functions like describe_data and analyze_categorical_columns from utils.py

Results
K-Means clustering with various k values was performed using the kmeans_clustering function in utils.py. This helped identify customer segments with distinct purchase behaviors.

# Utilities

## Data handling:
- ```load_data(file_path)```: Loads data from a file (CSV, Excel, JSON, etc.) into a pandas DataFrame.
- ```filter_data(df, col, valid_values=None, min_value=None, max_value=None)```: Filters data based on specific criteria in a column.

## Data exploration and analysis:

- ```describe_data(df, cols=None)```: Calculates and prints descriptive statistics for specified columns.
- ```analyze_categorical_columns(df, categorical_columns)```: Prints value counts and summary statistics for categorical columns.
- ```plot_sales(data, x_col, y_col, estimator=np.sum, title='', x_label='', y_label='')```: Creates a bar plot for sales data.


# License
This project is licensed under the MIT License.

# Contact

