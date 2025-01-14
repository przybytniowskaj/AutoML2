import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from mamut.wrapper import Mamut

iris = load_iris()


def test_multiclass():
    mamut = Mamut(n_iterations=1)
    X, y = pd.DataFrame(iris.data, columns=iris.feature_names), pd.DataFrame(
        iris.target, columns=["target"]
    )
    print(y.nunique())
    print(type(X), type(y))
    print(y.shape)
    mamut.fit(X, y)
    model = mamut.best_model_
    pred = model.predict(X)

    assert isinstance(pred, np.ndarray)


def test_binary():
    mamut = Mamut(n_iterations=1)
    X, y = (
        pd.DataFrame(iris.data, columns=iris.feature_names)[:100],
        pd.DataFrame(iris.target, columns=["target"])[:100],
    )
    print(y.nunique())
    print(type(X), type(y))
    print(y.shape)
    mamut.fit(X, y)
    model = mamut.best_model_
    pred = model.predict(X)

    assert isinstance(pred, np.ndarray)
