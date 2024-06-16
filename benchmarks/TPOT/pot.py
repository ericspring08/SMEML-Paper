import joblib
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import time

time_start = time.time()

data = pd.read_csv('../../benchmark_datasets/parkinsons.csv')

X = data.drop(columns=['status'])
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

tpot = TPOTClassifier(generations=50, population_size=20,
                      verbosity=2, random_state=42)

tpot.fit(X_train, y_train)

print("Test Accuracy:", tpot.score(X_test, y_test))

tpot.export('tpot_heart_pipeline.py')

# save model
joblib.dump(tpot.fitted_pipeline_, 'tpot_heart_model.pkl')

time_end = time.time()

time_total = time_end - time_start
print("Time taken: ", time_total)
