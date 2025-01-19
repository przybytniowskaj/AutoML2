import numpy as np

from mamut.wrapper import Mamut


def test_wrapper_binary_target(X, binary_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_multiclass_target(X, multiclass_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, multiclass_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_imbalanced_target(X, imbalanced_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, imbalanced_y)
    pred = mamut.best_model_.predict(X)

    n_before = X.shape[0]
    n_after = mamut.X_train.shape[0] + mamut.X_test.shape[0]

    assert n_after > n_before
    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_missing_data(X_missing, binary_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X_missing, binary_y)
    pred = mamut.best_model_.predict(X_missing)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_pca(X, binary_y):
    mamut = Mamut(n_iterations=1, pca=True)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_selection(X, binary_y):
    mamut = Mamut(n_iterations=1, feature_selection=True)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)

    assert mamut.best_score_ is not None
    assert isinstance(pred, np.ndarray)


def test_wrapper_subsequent_predict_calls(X, binary_y):
    mamut = Mamut(n_iterations=1)
    mamut.fit(X, binary_y)
    pred = mamut.best_model_.predict(X)
    pred2 = mamut.best_model_.predict(X)

    assert (pred == pred2).all()
    assert isinstance(pred, np.ndarray)
    assert isinstance(pred2, np.ndarray)
    assert mamut.best_score_ is not None
