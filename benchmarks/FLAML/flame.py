from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import xgboost

xgboost_version = xgboost.__version__

# Load your dataset
dataset_path = 'heart.csv'  # Specify the path to your dataset file
data = pd.read_csv(dataset_path)

X = data.drop(columns=["target"])  # Adjust the column name as per your dataset
y = data["target"]  # Adjust the column name as per your dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Customize FLAML's AutoML
automl = AutoML()
automl_settings = {
    "time_budget": 60,  # Set the maximum time budget for AutoML in seconds
    # Specify the type of task (classification or regression)
    "task": "classification",
    "metric": "accuracy",  # Choose the evaluation metric to optimize
    "learner_selector": "auto",  # Let FLAML automatically select the learners
    # Add more parameters as needed
}

# Fit the model using the customized settings
automl.fit(X_train, y_train, **automl_settings)

# Generate predictions using the trained model
y_pred = automl.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# save the model
with open('model.pkl', 'wb') as model_pkl:
    pickle.dump(automl, model_pkl)
