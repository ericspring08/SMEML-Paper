import os
import argparse as ap
import subprocess
from gen_dataset import gen_dataset
from get_attributes import get_attributes
import pandas as pd

datasets_folder = "../kaggle-dataset-download/DATASETS/"

# read the datasets folder
datasets = os.listdir(datasets_folder)

columns = ["num_rows", "num_cols", "output_distribution", "missing_values",
           "numerical_columns", "binary_categorical",
           "categorical_average_count", "average_iqr", "average_q1",
           "average_q3", "iqr_std", "q1_std", "q3_std",
           "z_score_average", "z_score_std", "average_correlation",
           "correlation_std", "dataset"]

attributes_df = pd.DataFrame(columns=columns)

count_target_by_name = 0
count_target_by_last = 0
combined = 0

for index, dataset in enumerate(datasets):
    if dataset.endswith('.csv'):
        # check if are column names
        try:
            df = pd.read_csv(datasets_folder + dataset)
            columns = df.columns

            possible_target_columns = ["target", "class", "label", "output", "y", "target_variable", "disease", "diag",
                                       "diagnosis", "result", "status", "outcome", "response", "response_variable", "dependent_variable", "dependent"]

            # lowercase every column name
            columns = [column.lower() for column in columns]

            target_column_val = "None"
            # check if there is a possible target column
            for target_column in possible_target_columns:
                if target_column in columns:
                    if df[target_column].nunique() == 2:
                        count_target_by_name += 1
                        target_column_val = df[target_column].unique()
                        break

            # check if last column is binary
            last_column = df.columns[-1]

            if df[last_column].nunique() == 2:
                count_target_by_last += 1

            print(
                f"Dataset {index + 1}/{len(datasets)}: {dataset} - {df.shape} - {target_column} - {count_target_by_last} - {count_target_by_name} - {combined}")

            # if either
            if any(target_column in columns for target_column in possible_target_columns) or df[last_column].nunique() == 2:
                combined += 1
        except:
            pass

print(f"Count target by last column: {count_target_by_last}")
print(f"Count target by name: {count_target_by_name}")
print(f"Combined: {combined}")
