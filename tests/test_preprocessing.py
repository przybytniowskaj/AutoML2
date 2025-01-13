import numpy as np
import pandas as pd

from mamut.preprocessing.preprocessing import Preprocessor

pd.set_option("display.max_columns", None)


def test_preprocessor():
    df = pd.DataFrame(
        {
            "num1": [1, 2.0, 3, np.nan, 5],
            "num2": [5.0, np.nan, 1, 3, 4],
            "cat1": ["A", "B", "A", "B", np.nan],
            "cat2": ["X", np.nan, "Y", "X", "Z"],
        }
    )

    df2 = df.copy()

    y = pd.Series([1, 0, 1, 0, 1])

    prp = Preprocessor(feature_selection=True, pca=True)
    X, y = prp.fit_transform(df, y)
    A = prp.transform(df2)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(A, np.ndarray)
