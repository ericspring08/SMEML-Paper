import os
import argparse as ap
import subprocess
from gen_dataset import gen_dataset
from get_attributes import get_attributes
import pandas as pd

datasets_folder = "../kaggle-dataset-download/DATASETS/"

# read the datasets folder
datasets = os.listdir(datasets_folder)

count_target_by_name = 0
count_target_by_last = 0
combined = 0

target_names = []

for index, dataset in enumerate(datasets):
    if dataset.endswith('.csv'):
        # check if are column names
        try:
            df = pd.read_csv(datasets_folder + dataset)

        except Exception as e:
            print("Error: ", e)
            continue
        if df.empty:
            continue

        columns = df.columns

        possible_target_columns = [
            "target",
            "class",
            "label",
            "output",
            "result",
            "score",
            "value",
            "response",
            "prediction",
            "predict",
            "outcome",
            "y",
            "yhat",
            "disease",
            "classification",
            "heart",
            "cancer",
            "diagnosis",
            "status",
            "grade",
            "rating",
            "death",
            "survival",
            "risk",
            "quality",
            "failure",
            "success",
            "pass",
            "fail",
            "positive",
            "negative",
            "win",
            "lose",
            "kidney",
            "diabetes",
            "renal",
            "chickenpox",
            "influenza",
            "congenital",
            "infection",
            "infect",
            "infectious",
            "virus",
            "bacteria"
        ]

        # lowercase every column name
        has_target_column = False
        for i, column in enumerate(columns):
            print("Column: ", column)
            if column.lower() in possible_target_columns and df[column].nunique() == 2:
                has_target_column = True
                target_names.append(column)
                break

        if has_target_column:
            count_target_by_name += 1

print("Count target by name: ", count_target_by_name)
print("Target names: ", target_names)
