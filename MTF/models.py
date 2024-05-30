from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import NuSVC
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
import numpy as np

classifiers = np.array([
    SVC(kernel='rbf', C=1.0, verbose=False),
    SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3),
    RidgeClassifierCV(alphas=[0.1, 1.0, 10.0]),
    RidgeClassifier(alpha=1.0),
    Perceptron(max_iter=1000, tol=1e-3),
    PassiveAggressiveClassifier(max_iter=1000, tol=1e-3),
    LogisticRegressionCV(cv=5, max_iter=1000, verbose=0),
    LogisticRegression(max_iter=1000, verbose=0),
    LinearSVC(max_iter=1000, verbose=0),
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
    DummyClassifier(strategy='most_frequent')
], dtype=object)
