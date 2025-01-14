import pandas as pd
import sklearn.datasets
from sklearn import clone
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from mamut.wrapper import Mamut

if __name__ == "__main__":

    print("Started")
    credit_g = sklearn.datasets.load_breast_cancer()
    X = credit_g.data
    X = pd.DataFrame(X)
    y = credit_g.target
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y)
    print("Loaded")

    mamut = Mamut(n_iterations=1, exclude_models=["SVC", "MLPClassifier"])
    print("Initialized")

    mamut.fit(X, y)
    print("Fitted")

    mamut.evaluate()

    # ensemble = mamut.create_greedy_ensemble(voting = "soft")
    # print("Ensemble created")
    # print(type(ensemble))
    # print(accuracy_score(mamut.y_test, ensemble.predict(mamut.X_test)))
    # print(ensemble.named_steps["model"].estimators)
    #
    # ensemble_clone = clone(ensemble)
    # print("Cloned")
    #
    # print("Fitting ensemble by hand")
    # ensemble_clone.fit(X_train, y_train)
    # y_pred = ensemble_clone.predict(X_test)
    # print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))


    # Check X_train and X_test from mamut
    X_train = mamut.X_train
    X_test = mamut.X_test
    y_train = mamut.y_train
    y_test = mamut.y_test


    # m = LogisticRegression()
    # print("Initialized")
    # print(y_train)
    # m.fit(X_train, y_train)
    # print("Fitted")
    # y_pred = m.predict(X_test)
    # print("ROC AUC Score: ", roc_auc_score(y_test, y_pred))






