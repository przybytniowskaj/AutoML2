import unittest

import numpy as np
import pandas as pd

from mamut.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data = (
            pd.DataFrame(
                {
                    "num1": [1, 2, 3, np.nan, 5],
                    "num2": [5, np.nan, 1, 3, 4],
                    "cat1": ["A", "B", "A", "B", np.nan],
                    "cat2": ["X", np.nan, "Y", "X", "Z"],
                }
            ),
            pd.Series([1, 0, 1, 0, 1]),
        )

    # def test_estimator_compliance(self):
    #     """Ensure the class complies with scikit-learn's estimator requirements."""
    #     check_estimator(DataPreprocessor())

    def test_init(self):
        """Test initialization of the DataPreprocessor."""
        preprocessor = DataPreprocessor(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1", "cat2"],
            num_imputation="mean",
            cat_imputation="most_frequent",
            feature_selection=True,
            pca=True,
            random_state=42,
        )

        self.assertEqual(preprocessor.num_imputation, "mean")
        self.assertEqual(preprocessor.cat_imputation, "most_frequent")
        self.assertTrue(preprocessor.feature_selection)
        self.assertTrue(preprocessor.pca)
        self.assertEqual(preprocessor.random_state, 42)

    def test_fit(self):
        """Test the fit method."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor(
            numeric_features=["num1", "num2"], categorical_features=["cat1", "cat2"]
        )
        preprocessor.fit(X, y)

        self.assertIsNotNone(preprocessor.pipeline_)

    def test_transform_before_fit(self):
        """Ensure transform cannot be called before fit."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor()
        with self.assertRaises(RuntimeError):
            preprocessor.transform(X, y)

    def test_fit_transform(self):
        """Test the fit_transform method."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor()
        X_transformed = preprocessor.fit_transform(X, y)

        self.assertGreater(X_transformed.shape[1], 0)

    def test_num_imputation(self):
        """Test numerical imputation strategies."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor(num_imputation="mean")
        preprocessor.fit(X, y)

        X_transformed = preprocessor.transform(X, y)
        self.assertFalse(np.any(np.isnan(X_transformed)))

    def test_cat_imputation(self):
        """Test categorical imputation strategies."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor(cat_imputation="most_frequent")
        preprocessor.fit(X, y)

        X_transformed = preprocessor.transform(X, y)
        self.assertFalse(pd.isnull(X_transformed).any().any())

    def test_feature_selection(self):
        """Test feature selection integration."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor(feature_selection=True, random_state=42)
        preprocessor.fit(X, y)

        self.assertTrue(
            any(step[0] == "feature_selection" for step in preprocessor._steps)
        )

    def test_pca(self):
        """Test PCA integration."""
        X, y = self.sample_data
        preprocessor = DataPreprocessor(pca=True, random_state=42)
        preprocessor.fit(X, y)

        self.assertTrue(any(step[0] == "pca" for step in preprocessor._steps))

    def test_invalid_num_imputation(self):
        """Test invalid numerical imputation strategy."""
        with self.assertRaises(KeyError):
            DataPreprocessor(num_imputation="invalid")

    def test_invalid_cat_imputation(self):
        """Test invalid categorical imputation strategy."""
        with self.assertRaises(KeyError):
            DataPreprocessor(cat_imputation="invalid")


if __name__ == "__main__":
    unittest.main()
