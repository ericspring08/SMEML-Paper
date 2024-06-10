import pandas as pd
import numpy as np
import argparse as ap
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from models import classifiers
import time
import os


def gen_dataset(input, output):
    # Load dataset
    try:
        df = pd.read_csv(input)

# check if dataset is greater than 10000
        if df.shape[0] > 2000:
            df = df.sample(2000)
# extract last column as target
        target = df.iloc[:, -1]
        df.drop(df.columns[-1], axis=1, inplace=True)

# convert target to binary
# get the unique values
        unique_values = target.unique()

# check if the target is binary
        if len(unique_values) > 2:
            print('Target is not binary')
            return "Target is not binary"
        else:
            # convert to binary
            target = target.apply(lambda x: 1 if x == unique_values[0] else 0)

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
