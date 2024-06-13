from SMEML.SMEML import SMEML
from sklearn.model_selection import train_test_split
import pandas as pd

# load the dataset
df = pd.read_csv('./datasets/heart.csv')

X = df.drop('target', axis=1)
y = df['target']

# label encoding
# y = y.map({'ckd': 1, 'notckd': 0})

# create an instance of the SMEML class
# X and y are the features and target variable
# respectively
smeml = SMEML()
# train the model
results, attributes = smeml.train(X, y)
