import numpy as np

from mamut.preprocessing.preprocessing import Preprocessor


def test_preprocessor_binary_y(X, binary_y):
    prp = Preprocessor(feature_selection=True, pca=True)
    Xft, yft = prp.fit_transform(X, binary_y)
    Xt = prp.transform(X)

    assert isinstance(Xft, np.ndarray)
    assert isinstance(yft, np.ndarray)
    assert isinstance(Xt, np.ndarray)


def test_subsequent_transform_calls(X, binary_y):
    prp = Preprocessor()
    Xft, yft = prp.fit_transform(X, binary_y)
    Xt = prp.transform(X)
    Xt2 = prp.transform(X)

    assert (Xt == Xt2).all().all()
    assert isinstance(Xt2, np.ndarray)
    assert isinstance(Xft, np.ndarray)
    assert isinstance(yft, np.ndarray)
    assert isinstance(Xt, np.ndarray)
