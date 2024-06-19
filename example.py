from SMEML.SMEML import SMEML
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import re

if __name__ == "__main__":
    time_start = time.time()
    # load the dataset
    df = pd.read_csv('./benchmark_datasets/heart2.csv')

    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    # replace No, borderline diabetes with borderline diabetes
    df['Diabetic'] = df['Diabetic'].replace('No, borderline diabetes', 'borderline diabetes')

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

# label encoding
    y = y.map({'Yes': 1, 'No': 0})

# create an instance of the SMEML class
# X and y are the features and target variable
# respectively
    smeml = SMEML(iterations=1, mode="SME")
# train the model
    smeml.train(X, y)
    time_end = time.time()
    print('Time taken to train the model: ', time_end - time_start)
