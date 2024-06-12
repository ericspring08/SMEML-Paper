import pandas as pd
import numpy as np
import argparse as ap
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from models import classifiers
import time
import os

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


def gen_dataset(input, output):
    # Load dataset
    try:
        df = pd.read_csv(input)

        columns = df.columns
        if len(columns) <= 2:
            print('Dataset has less than 2 columns')
            return "Dataset has less than 2 columns"

        has_target_column = False
        target_column = ""
        for i, column in enumerate(columns):
            print("Column: ", column)
            if column.lower() in possible_target_columns and df[column].nunique() == 2:
                target_column = column
                has_target_column = True
                break

        if not has_target_column:
            print('No target column found')
            return "No target column found"

        target = df[target_column]
        unique_values = target.unique()

        target = target.apply(lambda x: 1 if x == unique_values[0] else 0)

        if df.shape[0] > 2000:
            df = df.sample(2000)

        df.drop(df.columns[-1], axis=1, inplace=True)

        numerical_features = []
        categorical_features = []

# Loop through all columns
        for col in df.columns:
            # Check if column is numeric
            # check the number of unique values
            if df[col].nunique() < 15 or df[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        scaler = MinMaxScaler()

# replace with mode for unknown values
        df = df.fillna(df.mode().iloc[0])

        preprocessor = ColumnTransformer(
            transformers=[('ohe',
                           OneHotEncoder(handle_unknown='ignore',
                                         sparse_output=False),
                           categorical_features),
                          ('scaler',
                           scaler,
                           numerical_features)],
            remainder='passthrough',
            verbose_feature_names_out=False).set_output(transform='pandas')

        df = preprocessor.fit_transform(df)

# split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df, target, test_size=0.2)

# performance
        results = {}

        for clf in classifiers:
            try:
                time_start = time.time()
                clf = clf.fit(X_train, y_train)
                time_end = time.time()
                perf = clf.score(X_test, y_test)
                time_elapsed = time_end - time_start
                results[clf.__class__.__name__] = {
                    'score': perf, 'time': time_elapsed
                }
                print(
                    f'[{clf.__class__.__name__}] score: {perf} time: {time_elapsed}')
            except Exception as e:
                results[clf.__class__.__name__] = np.nan
                print(f'[{clf.__class__.__name__}] failed')

# create output directory if it does not exist
        if not os.path.exists(output):
            os.makedirs(output)
# convert to dataframe
        results = pd.DataFrame(results).T
# column names
        results.columns = ['score', 'time']
# set index name
        results.index.name = 'classifier'
        results.to_csv(os.path.join(output, 'results.csv'))
# save dataset with target column
        df['target'] = target
        df.to_csv(os.path.join(output, 'dataset.csv'), index=False)

        print('Dataset generated successfully')
        print(f'Dataset saved to {output}')
    except Exception as e:
        return str(e)
