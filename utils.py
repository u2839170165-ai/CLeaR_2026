import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from functools import reduce
import matplotlib.pyplot as plt

# Tools to save/load results    
def save_pickle(obj, path: str):
    path = Path(path)
    with path.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    path = Path(path)
    with path.open('rb') as f:
        return pickle.load(f)

def make_heom_metric(continuous_cols, discrete_cols, observed_cols, df):
    def custom_heom_metric(x, y):
        distance = 0
        for i, col in enumerate(observed_cols):
            if col in continuous_cols:
                denom = np.max([np.max(df[col]) - np.min(df[col]),1e-6])
                distance += ((np.abs(x[i] - y[i])) / denom)**2
            elif ((col in discrete_cols) and (x[i] != y[i])):
                distance += 1
        return distance**0.5
    return custom_heom_metric

def heterogeneous_knn_imputation(data, k = 100, n_neighbors_used = 3, cols_to_impute = None,
                                 continuous_cols = ['Age', 'SPY', 'TMB'], 
                                 discrete_cols = ['Sex', 'Smoking', 'Stage', 'Status'],
                                 out_dir=None, file_name=''):
    """
    Heterogeneous imputation based on kNN.
    
    Args:
        data: Input DataFrame
        k: Total number of neighbors to search (must be ≥ n_neighbors_used)
        n_neighbors_used: Number of neighbors to use for imputation
        cols_to_impute: Columns to impute (all by default)
        continuous_cols: List of continuous columns
        discrete_cols: List of discrete columns
        out_dir: Directory to save diagnostic plots
        file_name: Prefix for diagnostic files
    """

    if k < n_neighbors_used:
        raise ValueError(f"k ({k}) doit être ≥ n_neighbors_used ({n_neighbors_used})")
    
    all_cols = data.columns.tolist()
    df_all = data.copy()
    if cols_to_impute is None: cols_to_impute = all_cols
    df = data.copy()[cols_to_impute]
    df_imputed = df.copy()

    #Print total number of samples considered for imputation
    total_samples = df.shape[0]
    print(f"Total samples: {total_samples}")

    if out_dir:
        out_dir = Path(out_dir)
        # mssing fraction by column (among column to impute)
        miss_frac = df.isna().mean()
        # sort for readability
        miss_frac = miss_frac.sort_values(ascending=False)
        # adaptative figure
        fig_w = max(6, len(miss_frac) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_w, 4))
        miss_frac.plot(kind='bar', ax=ax, color='gray')
        ax.set_ylabel('Missing fraction')
        # ax.set_ylim(0, 1)
        # ax.set_title('Missing data prevalence per column')
        plt.xticks(rotation=45, ha='right')
        fname_png = (file_name + '_missing_prevalence.png') if file_name else 'missing_prevalence.png'
        fname_pdf = (file_name + '_missing_prevalence.pdf') if file_name else 'missing_prevalence.pdf'
        plt.savefig(out_dir / fname_png, bbox_inches='tight')
        plt.savefig(out_dir / fname_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved missing data prevalence plot to {out_dir / fname_png}")

    for idx, row in df[df.isna().any(axis=1)].iterrows():
        # availables columns for this row
        observed_cols = row[~row.isna()].index.tolist()
        missing_cols = row[row.isna()].index.tolist()

        # build complete rows for this column
        df_complete = df.dropna(subset=observed_cols)

        if df_complete.empty:
            continue 

        # Fit kNN only on observed columns
        heom_metric = make_heom_metric(continuous_cols, discrete_cols, observed_cols, df)
        knn = NearestNeighbors(n_neighbors=k, metric=heom_metric)
        knn.fit(df_complete[observed_cols].to_numpy())

        # find neighbors for this row
        distances, indices = knn.kneighbors([row[observed_cols].to_numpy()])
        
        
        # Get neighbors data
        neighbors = df_complete.iloc[indices[0]].copy()
        neighbors['distance'] = distances[0]
        
        # Imputation by data type
        for col in missing_cols:
            valid_neighbors = neighbors.dropna(subset=[col])
            n_available = len(valid_neighbors)
            if n_available == 0:
                print(f"No valid neighbors to impute column '{col}' for index {idx}")
                continue

            # Number of neighbors to use for this column
            n_to_use = min(n_neighbors_used, n_available)
            if n_available < n_neighbors_used:
                print(f"Only {n_available} valid neighbors available to impute column '{col}' for index {idx}, needed {n_neighbors_used}")

            closest_neighbors = valid_neighbors.nsmallest(n_to_use, 'distance')
            closest_distances = closest_neighbors['distance'].values

            if col in continuous_cols:
                # Weighted average by the inverse of the distance for the n nearest valid neighbors.
                weights = 1 / (closest_distances + 1e-6)  # Avoid dividing by zero
                weights = weights / weights.sum()  # weights normalization
                value = (closest_neighbors[col] * weights).sum()
            elif col in discrete_cols:
                # Take most frequent value
                value = closest_neighbors[col].value_counts().idxmax()
            else:
                continue
            df_imputed.at[idx, col] = value
        df_all.loc[idx, cols_to_impute] = df_imputed.loc[idx, cols_to_impute]
    if out_dir:
        cols_to_plot = continuous_cols+discrete_cols
        fig, axs = plt.subplots(2, len(cols_to_plot), figsize = (30, 20))
        for i, col in enumerate(cols_to_plot):
            ax1 = axs[0, i]
            ax2 = axs[1, i]
            ax1.hist(df[col].dropna(), alpha=0.5, color='blue')
            ax2.hist(df_imputed[col], alpha=0.5, color='orange')
            ax1.set_title(col, fontsize=23)
            ax2.set_title(col, fontsize=23)
            
            ax1.tick_params(axis='both', which='major', labelsize=15)
            ax2.tick_params(axis='both', which='major', labelsize=15)
        fname_png = file_name+".png"
        fname_pdf = file_name+".pdf"
        plt.savefig(out_dir/fname_png, bbox_inches='tight')
        plt.savefig(out_dir/fname_pdf, bbox_inches='tight')
        print(f"Saved {fname_png} and {fname_pdf}")
    return df_all

def merge_dataframes(data_list, merge_keys=['PatientID']):
    """
    Merges a list of pandas DataFrames based on multiple shared columns (keys),
    preserving all shared and non-shared columns and keeping one row per unique 
    combination of the merge keys.

    Args:
        data_list (list): A list of pandas DataFrames.
        merge_keys (list): A list of column names to use as the join key (e.g., ['PatientID', 'ListID']).

    Returns:
        pandas.DataFrame: The single merged DataFrame.
    """

    if not data_list:
        return pd.DataFrame()

    # The outer merge is performed using the list of merge_keys.
    # 'how=outer' still ensures all unique combinations of the keys are preserved.
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'),
        data_list
    )

    return df_merged

def selective_fillna(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    """
    Replaces NaN values with 0 in all columns EXCEPT those specified 
    in the exclude_cols list.

    Args:
        df (pd.DataFrame): The input DataFrame.
        exclude_cols (list): A list of column names where NaN should be preserved.

    Returns:
        pd.DataFrame: The DataFrame with NaNs selectively replaced.
    """
    
    # 1. Identify columns to FILL with 0
    # Get all column names from the DataFrame
    all_cols = set(df.columns)
    
    # Exclude the specified columns to get the list of columns to fill
    cols_to_fill = list(all_cols - set(exclude_cols))
    
    # 2. Perform the fill operation
    
    # Create a dictionary mapping the columns to be filled to the value 0
    fill_dict = {col: 0 for col in cols_to_fill}
    
    # Use .fillna() on the entire DataFrame with the specific dictionary.
    # This only affects the columns explicitly named in the dictionary.
    df_filled = df.fillna(fill_dict)
    
    return df_filled

def str_to_class(df, mappings=None, categorical_cols=None, set_index=None, inplace=False):
    """
    Convert string columns to integer class codes (nullable Int64) using mapping dicts,
    optionally convert some columns to pandas.Categorical, and optionally set an index column.
    Returns a DataFrame (modified copy unless inplace=True).
    """
    if not inplace:
        df = df.copy()
    mappings = mappings or {}
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype('Int64')  # nullable integer dtype
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
    if set_index and set_index in df.columns:
        df = df.set_index(set_index)
    return df

def discretize_dataframe(df: pd.DataFrame, cols_to_discretize: list = None, n_bins: int = 10) -> pd.DataFrame:
    """
    Discretizes each column of a DataFrame into a specified number of bins using quartiles.

    Args:
        df: The input Pandas DataFrame.
        n_bins: The number of bins (quantiles) to use for discretization.
                For quartiles, n_bins=4.
        cols_to_discretize: List of column names to discretize. If None, all columns are discretized.

    Returns:
        A new DataFrame with all columns discretized.
    """
    # By default, discretize only numeric columns to avoid passing strings/objects
    if cols_to_discretize is None:
        cols_to_discretize = df.select_dtypes(include=[np.number]).columns.tolist()

    discretized_df = df.copy()
    for col in cols_to_discretize:
        # Safely coerce to numeric (preserve NaNs for non-convertible values)
        series = pd.to_numeric(df[col], errors='coerce')

        # If after coercion the series is all-NaN or has fewer than 2 unique values,
        # skip discretization and leave the column as-is (or set to NaN-coded)
        non_na = series.dropna()
        if non_na.shape[0] < 2:
            # Not enough data to form bins; copy the coerced series back (keeps NaNs)
            discretized_df[col] = series
            continue

        # Use qcut for quantile-based discretization; catch possible ValueError
        try:
            discretized = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
            discretized_df[col] = discretized
        except ValueError as e:
            # e.g., all values equal leading to ValueError: Bin edges must be unique
            # Fallback: map to a single bin (0) for non-NaN values
            discretized_df[col] = series.notna().astype('Int64') - 1
    return discretized_df