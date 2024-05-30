import pandas as pd
import numpy as np
import argparse as ap
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from models import classifiers
import time
import os

# Parse arguments
parser = ap.ArgumentParser(description='Generate dataset')
parser.add_argument('-i', '--input', type=str,
                    help='input_dataset', required=True)
parser.add_argument('-o', '--output', type=str,
                    help='output_dataset', required=True)

args = parser.parse_args()

# create output directory if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Load dataset
df = pd.read_csv(args.input)

# limit to 10K rows
if len(df) > 10000:
    df = df.sample(n=10000)

# extract last column as target
target = df.iloc[:, -1]
df.drop(df.columns[-1], axis=1, inplace=True)

# convert target to binary
# get the unique values
unique_values = target.unique()

# check if the target is binary
if len(unique_values) > 2:
    raise ValueError('Target must be binary')
else:
    # convert to binary
    target = target.apply(lambda x: 1 if x == unique_values[0] else 0)

numerical_features = []
categorical_features = []

# Loop through all columns
for col in df.columns:
    # Check if column is numeric
    # check the number of unique values
    if df[col].nunique() > 15:
        numerical_features.append(col)
    else:
        categorical_features.append(col)

scaler = StandardScaler()

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
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)

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
        print(f'[{clf.__class__.__name__}] score: {perf} time: {time_elapsed}')
    except Exception as e:
        print(e)
        results[clf.__class__.__name__] = np.nan

# convert to dataframe
results = pd.DataFrame(results).T
# column names
results.columns = ['score', 'time']
# set index name
results.index.name = 'classifier'
results.to_csv(os.path.join(args.output, 'results.csv'))
# save dataset with target column
df['target'] = target
df.to_csv(os.path.join(args.output, 'dataset.csv'), index=False)

print('Dataset generated successfully')
print(f'Dataset saved to {args.output}')
