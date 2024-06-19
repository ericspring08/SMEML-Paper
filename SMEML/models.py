from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
import numpy as np

classifiers = np.array([
    SVC(kernel='rbf', C=1.0, verbose=False),
    SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3),
    RidgeClassifier(alpha=1.0),
    Perceptron(max_iter=1000, tol=1e-3),
    PassiveAggressiveClassifier(max_iter=1000, tol=1e-3),
    LogisticRegression(max_iter=1000, verbose=0),
    LinearSVC(max_iter=1000, dual=True, verbose=0),
    RandomForestClassifier(n_estimators=100, verbose=0),
    HistGradientBoostingClassifier(max_iter=100, verbose=0),
    GradientBoostingClassifier(n_estimators=100, verbose=0),
    ExtraTreesClassifier(n_estimators=100, verbose=0),
    AdaBoostClassifier(n_estimators=100),
    XGBClassifier(n_estimators=100, verbosity=0),
    LGBMClassifier(n_estimators=100, verbose=-1),
    CatBoostClassifier(iterations=100, silent=True),
    RadiusNeighborsClassifier(radius=1.0),
    KNeighborsClassifier(n_neighbors=5),
    NearestCentroid(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    GaussianNB(),
    BernoulliNB(alpha=1.0),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=False),
    ExtraTreeClassifier(),
    DecisionTreeClassifier(),
    LabelSpreading(kernel='rbf'),
    LabelPropagation(kernel='rbf'),
    DummyClassifier(strategy='most_frequent'),
], dtype=object)


param_grids = {
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
    },
    'SGDClassifier': {
        'loss': ['squared_error', 'squared_epsilon_insensitive', 'squared_hinge', 'hinge', 'perceptron', 'huber', 'epsilon_insensitive', 'log_loss', 'modified_huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'max_iter': [1000, 2000],
    },
    'RidgeClassifier': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    },
    'Perceptron': {
        # 'penalty': [None, 'l2', 'l1', 'elasticnet'],
        # 'alpha': [0.001, 0.01, 0.1],
        'max_iter': [1000, 2000],
        # 'tol': [1e-3, 1e-4],
    },
    'PassiveAggressiveClassifier': {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [1000, 2000],
        'tol': [1e-3, 1e-4],
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [1000, 2000],
    },
    'LinearSVC': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'loss': ['hinge', 'squared_hinge'],
        'max_iter': [1000, 2000],
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    },
    'HistGradientBoostingClassifier': {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [100, 200],
        'max_leaf_nodes': [31, 64],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [20, 50],
    },
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 1.0],
    },
    'ExtraTreesClassifier': {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
    },
    'XGBClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    },
    'LGBMClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20],
        'num_leaves': [31, 64],
    },
    'CatBoostClassifier': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7],
    },
    'RadiusNeighborsClassifier': {
        'radius': [0.5, 1.0, 2.0],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
    'NearestCentroid': {
        'metric': ['euclidean', 'manhattan'],
    },
    'QuadraticDiscriminantAnalysis': {
        'reg_param': [0.0, 0.1, 0.5, 1.0],
    },
    'LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.5, 1.0],
    },
    'GaussianNB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7],
    },
    'BernoulliNB': {
        'alpha': [0.01, 0.1, 1.0],
        'binarize': [0.0, 0.5, 1.0],
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(50,), (100,), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [1000, 2000],
    },
    'ExtraTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'LabelSpreading': {
        'gamma': [0.1, 1, 10],
        'n_neighbors': [3, 5, 7],
    },
    'LabelPropagation': {
        'gamma': [0.1, 1, 10],
        'n_neighbors': [3, 5, 7],
    },
    'DummyClassifier': {
        'strategy': ['most_frequent', 'stratified', 'uniform', 'constant'],
    }
}
