import os
import argparse as ap
import subprocess
from gen_dataset import gen_dataset
from get_attributes import get_attributes
import pandas as pd

parser = ap.ArgumentParser(description='Generate datasets')

parser.add_argument('-f',
                    '--folder',
                    type=str,
                    help='Folder containing datasets',
                    required=True)

parser.add_argument('-o',
                    '--output',
                    type=str,
                    help='Output folder',
                    required=True)

args = parser.parse_args()

datasets_folder = args.folder
output_folder = args.output

# read the datasets folder
datasets = os.listdir(datasets_folder)

columns = ["num_rows", "num_cols", "output_distribution",
           "numerical_columns", "binary_categorical",
           "categorical_average_count", "average_iqr", "average_q1",
           "average_q3", "iqr_std", "q1_std", "q3_std",
           "z_score_average", "z_score_std", "average_correlation",
           "correlation_std", "dataset"]

attributes_df = pd.DataFrame(columns=columns)


for index, dataset in enumerate(datasets):
    if dataset.endswith('.csv'):
        print(f"[{index}] Generating {dataset}")
        # print subprocess output
        error_ = gen_dataset(os.path.join(datasets_folder, dataset),
                             os.path.join(output_folder, dataset.split('.')[0]))

        if error_:
            print(f"Error: {error_}")
            continue

        attributes = get_attributes(os.path.join(
            output_folder, dataset.split('.')[0], 'dataset.csv'))

        attributes['dataset'] = dataset

        attributes_df = attributes_df._append(attributes, ignore_index=True)

attributes_df.to_csv('attributes.csv', index=False)
