# sklearn intel patch
# check if x86 or arm
import platform
if platform.machine() == 'x86_64':
    from sklearnex import patch_sklearn
    patch_sklearn()

import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, VotingClassifier
from skopt import BayesSearchCV
from xgboost import XGBClassifier
import os
import multiprocessing
from SMEML.models import classifiers, param_grids
from functools import partial
import logging

np.int = int

logging.captureWarnings(capture=True)

# Get logger for warnings
logger = logging.getLogger("py.warnings")

# StreamHandler outputs on sys.stderr by default
handler = logging.StreamHandler()
logger.addHandler(handler)

warnings_to_ignore = [
    "FutureWarning",
    "InconsistentVersionWarning"
]

# Set rule to ignore warnings
logger.addFilter(
    lambda record: record.getMessage() in warnings_to_ignore)

classifier_names = [
    'SVC',
    'SGDClassifier',
    'RidgeClassifier',
    'Perceptron',
    'PassiveAggressiveClassifier',
    'LogisticRegression',
    'LinearSVC',
    'RandomForestClassifier',
    'HistGradientBoostingClassifier',
    'GradientBoostingClassifier',
    'ExtraTreesClassifier',
    'AdaBoostClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'CatBoostClassifier',
    'RadiusNeighborsClassifier',
    'KNeighborsClassifier',
    'NearestCentroid',
    'QuadraticDiscriminantAnalysis',
    'LinearDiscriminantAnalysis',
    'GaussianNB',
    'BernoulliNB',
    'MLPClassifier',
    'ExtraTreeClassifier',
    'DecisionTreeClassifier',
    'LabelSpreading',
    'LabelPropagation',
    'DummyClassifier'
]


class SMEML:
    def __init__(self, iterations=10):
        self.X = None
        self.y = None
        self.iterations = iterations

    def train(self, X, y):
        self.X = X
        self.y = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # fill nan values
        self.preprocess()

        attributes = self.get_dataset_attributes()

        # train the model
        smeml_model = pkl.load(
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl'), 'rb'))

        input = list(attributes.values())
        input = np.array(input).reshape(1, -1)

        results = smeml_model.predict(input)

        top = np.argsort(results[0])[-8:]

        model_threads = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for i in top:
            thread = multiprocessing.Process(name=classifier_names[i],
                target=self.train_thread, args=(classifiers[i], classifier_names[i], return_dict,))
            model_threads.append(thread)

        for thread in model_threads:
            thread.start()

        # wait for all threads to finish
        for thread in model_threads:
            thread.join()
        
        self.model_accuracies = []
        self.models = []

        # loop through return dict and get the models and accuracies
        for key, value in return_dict.items():
            if 'accuracy' in key:
                self.model_accuracies.append((key, value))
            else:
                self.models.append((key.replace('_model', ''), value))

        print(self.models)

        stacking_classifier = StackingClassifier(estimators=self.models, final_estimator=XGBClassifier())

        stacking_classifier.fit(self.X_train, self.y_train)

        accuracy = stacking_classifier.score(self.X_test, self.y_test)

        print("stacking classifier accuracy: ", accuracy)

        self.best_model = (stacking_classifier, accuracy)

        self.generate_report()
        self.save_final_model()

    def train_thread(self, model, model_name, return_dict):
        print("Training model: ", model_name)
        optimizer = BayesSearchCV(
            model, param_grids[model_name], n_iter=self.iterations, cv=3, error_score=0)

        optimizer.fit(self.X_train, self.y_train,
                      callback=partial(self.bayes_cv_callback, model_name=model_name))

        accuracy = optimizer.score(self.X_test, self.y_test)
        print("Model: ", model, " with accuracy: ", accuracy)


        return_dict[model_name + '_accuracy'] = accuracy
        return_dict[model_name + '_model'] = optimizer.best_estimator_

    def bayes_cv_callback(self, res, model_name=None):
        # print the current model name
        print(f"[{model_name}] Iteration {res['x']}, accuracy: {res['fun']*-1}")

    def get_dataset_attributes(self):
        attributes = {}

        attributes['num_rows'] = self.X.shape[0]
        print("Number of rows: ", self.X.shape[0])
        attributes['num_cols'] = self.X.shape[1]
        print("Number of columns: ", self.X.shape[1])

        # get the output distribution of 1 percentage
        attributes['output_distribution'] = max(self.X.iloc[:, -
                                                            1].value_counts(normalize=True))
        print("Output distribution: ", max(
            self.X.iloc[:, -1].value_counts(normalize=True)))

        # get percentage of numerical and categorical columns
        numerical_features = 0
        binary_categorical = 0
        categorical_average_count = 0
        categorical_count = 0
        iqr_values = []
        q1_values = []
        q3_values = []
        for col in self.X.columns:
            if self.X[col].nunique() > 15:
                numerical_features += 1

                # get the interquartile range
                q1 = self.X[col].quantile(0.25)
                q3 = self.X[col].quantile(0.75)
                iqr = q3 - q1

                iqr_values.append(iqr)
                q1_values.append(q1)
                q3_values.append(q3)
            else:
                categorical_count += 1
                # categorical
                # check if binary or not
                if self.X[col].nunique() == 2:
                    binary_categorical += 1

                categorical_average_count += self.X[col].nunique()

        attributes['numerical_columns'] = numerical_features/self.X.shape[1]
        attributes['binary_categorical'] = binary_categorical/categorical_count
        attributes['categorical_average_count'] = categorical_average_count / \
            categorical_count
        attributes['average_iqr'] = np.mean(iqr_values)
        attributes['average_q1'] = np.mean(q1_values)
        attributes['average_q3'] = np.mean(q3_values)
        attributes['iqr_std'] = np.mean(np.std(iqr_values))
        attributes['q1_std'] = np.mean(np.std(q1_values))
        attributes['q3_std'] = np.mean(np.std(q3_values))
        print("Numerical columns: ", numerical_features/self.X.shape[1])
        print("Binary categorical: ", binary_categorical/categorical_count)
        print("Categorical average count: ",
              categorical_average_count/categorical_count)
        print("Average IQR: ", np.mean(iqr_values))
        print("Average Q1: ", np.mean(q1_values))
        print("Average Q3: ", np.mean(q3_values))
        print("IQR STD: ", np.mean(np.std(iqr_values)))
        print("Q1 STD: ", np.mean(np.std(q1_values)))
        print("Q3 STD: ", np.mean(np.std(q3_values)))

        # average percentage of values with a z-score greater than 3 in each dataset
        z_score_average = np.mean(
            np.abs((self.X - self.X.mean())/self.X.std()))
        z_score_std = np.mean(
            np.std(np.abs((self.X - self.X.mean())/self.X.std())))

        attributes['z_score_average'] = z_score_average
        print("Z-score average: ", z_score_average)
        attributes['z_score_std'] = z_score_std
        print("Z-score std: ", z_score_std)

        if self.X.shape[0] * self.X.shape[1] > 100000:
            attributes['average_correlation'] = np.nan
            attributes['correlation_std'] = np.nan
        else:
            # correlation matrix of the dataset
            corr_matrix = self.X.corr()

            # get the average correlation of the dataset (excluding the diagonal values)
            corr_values = []

            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    if i != j:
                        corr_values.append(corr_matrix.iloc[i, j])

            attributes['average_correlation'] = np.mean(corr_values)
            print("Average correlation: ", np.mean(corr_values))
            attributes['correlation_std'] = np.mean(np.std(corr_values))
            print("Correlation std: ", np.mean(np.std(corr_values)))

        self.attributes = attributes

        return attributes

    def preprocess(self):
        columns = self.X.columns

        categorical_features = self.X.select_dtypes(include=['object']).columns
        numerical_features = self.X.select_dtypes(
            include=['int64', 'float64']).columns

        # fill missing values
        for col in columns:
            if col in categorical_features:
                self.X[col] = self.X[col].fillna(
                    self.X[col].mode()[0])
            else:
                self.X[col] = self.X[col].fillna(
                    self.X[col].mean())

        # scale
        scaler = MinMaxScaler()

# replace with mode for unknown values
        df = self.X.fillna(self.X.mode().iloc[0])

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

        self.X = preprocessor.fit_transform(self.X)

    def generate_report(self):
        lines = []
        lines.append("Dataset information")

        print()

        for key, value in self.attributes.items():
            lines.append(f"{key}: {value}")

        print()

        lines.append("Top 10 classifiers predicted by SMEML")

        for index, model in enumerate(self.models):
            lines.append("Model " + str(index) + ": " + str(model[0]) +
                         " with accuracy: " + str(model[1]))

        print()

        lines.append("Best model" + str(self.best_model[0]) +
                     " with accuracy average: " +
                     str(self.best_model[1]))

        # save the report
        with open('report.txt', 'w') as f:
            for line in lines:
                f.write("%s\n" % line)

    def save_final_model(self):
        pkl.dump(self.best_model[0], open('best_model.pkl', 'wb'))
