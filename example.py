from SMEML.SMEML import SMEML
from sklearn.model_selection import train_test_split
import pandas as pd
import time

if __name__ == "__main__":
    time_start = time.time()
    # load the dataset
    df = pd.read_csv('./benchmark_datasets/heart2.csv')

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

# label encoding
# y = y.map({'ckd': 1, 'notckd': 0})

# create an instance of the SMEML class
# X and y are the features and target variable
# respectively
    smeml = SMEML(iterations=10)
# train the model
    smeml.train(X, y)
    time_end = time.time()
    print('Time taken to train the model: ', time_end - time_start)
