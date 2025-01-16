import numpy as np

from mamut.preprocessing.preprocessing import Preprocessor
from tests.mock import X, binary_y


def test_preprocessor_binary_y(X, binary_y):
    prp = Preprocessor(feature_selection=True, pca=True)
    Xft, yft = prp.fit_transform(X, binary_y)
    Xt = prp.transform(X)

    assert isinstance(Xft, np.ndarray)
    assert isinstance(yft, np.ndarray)
    assert isinstance(Xt, np.ndarray)
