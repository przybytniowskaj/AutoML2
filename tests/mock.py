import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def X():
    num1 = np.random.normal(0, 1, 100)
    num2 = np.random.randint(1, 100, 100)
    cat1 = np.random.choice(["A", "B", "C"], 100)
    cat2 = np.random.choice(["X", "Y", "Z"], 100)

    return pd.DataFrame({"num1": num1, "num2": num2, "cat1": cat1, "cat2": cat2})


@pytest.fixture
def X_missing():
    num1 = np.random.normal(0, 1, 100)
    num2 = np.random.randint(1, 100, 100).astype(float)
    cat1 = np.random.choice(["A", "B", "C"], 100)
    cat2 = np.random.choice(["X", "Y", "Z"], 100)

    missing1_ids = np.random.choice(range(100), 10)
    missing2_ids = np.random.choice(range(100), 10)
    missing3_ids = np.random.choice(range(100), 10)
    missing4_ids = np.random.choice(range(100), 10)

    num1[missing1_ids] = np.nan
    num2[missing2_ids] = np.nan
    cat1[missing3_ids] = np.nan
    cat2[missing4_ids] = np.nan

    return pd.DataFrame({"num1": num1, "num2": num2, "cat1": cat1, "cat2": cat2})


@pytest.fixture
def binary_y():
    return pd.Series(np.repeat([0, 1], [50, 50]))


@pytest.fixture
def multiclass_y():
    return pd.Series(np.repeat([0, 1, 2], [34, 33, 33]))


@pytest.fixture
def imbalanced_y():
    return pd.Series(np.repeat([0, 1], [92, 8]))
