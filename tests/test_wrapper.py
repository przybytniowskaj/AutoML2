import pytest
from sklearn.datasets import fetch_openml

from mamut.wrapper import Mamut


@pytest.fixture
def credit_data():
    credit_g = fetch_openml(data_id=31)
    X = credit_g.data
    y = credit_g.target
    return X, y


@pytest.fixture
def mamut_instance():
    return Mamut(n_iterations=1)


def test_mamut_fit(credit_data, mamut_instance):
    model = mamut_instance.fit(*credit_data)
    assert model is not None
    assert mamut_instance.fitted_models_ is not None
    assert mamut_instance.best_model_ is not None
    assert mamut_instance.best_score_ is not None
    assert mamut_instance.results_df_ is not None
