from flaml import AutoML
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
dataset_path = 'heart.csv'  # Specify the path to your dataset file
data = pd.read_csv(dataset_path)d

X = data.drop(columns=["target"])  # Adjust the column name as per your dataset
y = data["target"]  # Adjust the column name as per your dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Customize FLAML's AutoML
automl = AutoML()
automl_settings = {
    "time_budget": 200,  # Set the maximum time budget for AutoML in seconds
    "task": "classification",  # Specify the type of task (classification or regression)
    "metric": "accuracy",  # Choose the evaluation metric to optimize
    "learner_selector": "auto",  # Let FLAML automatically select the learners
    "ensemble_size": 5,  # Set the size of the ensemble model
    # Add more parameters as needed
}

# Fit the model using the customized settings
automl.fit(X_train, y_train, **automl_settings)

# Generate predictions using the trained model
y_pred = automl.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
