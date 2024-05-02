from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('heart.csv')

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

tpot.fit(X_train, y_train)

print("Test Accuracy:", tpot.score(X_test, y_test))

tpot.export('tpot_heart_pipeline.py')
