from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    imputer_mapping = {
        "iterative": IterativeImputer,
        "knn": KNNImputer,
        "mean": lambda: SimpleImputer(strategy="mean"),
        "median": lambda: SimpleImputer(strategy="median"),
        "constant": lambda: SimpleImputer(strategy="constant", fill_value=0),
    }

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        num_imputation: Literal[
            "iterative", "mean", "median", "constant"
        ] = "iterative",
        cat_imputation: Literal["most_frequent", "constant"] = "constant",
        feature_selection: bool = False,
        pca: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.num_imputation = num_imputation
        self.cat_imputation = cat_imputation
        self.feature_selection = feature_selection
        self.pca = pca
        self.random_state = random_state

        self._num_steps = None
        self._cat_steps = None
        self._steps = None
        self.pipeline_ = None

        if self.num_imputation not in self.imputer_mapping:
            raise KeyError(f"Invalid num_imputation strategy: {self.num_imputation}")
        if self.cat_imputation not in ["most_frequent", "constant"]:
            raise KeyError(f"Invalid cat_imputation strategy: {self.cat_imputation}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._steps = []
        self._num_steps = [
            ("imputer", self.imputer_mapping[self.num_imputation]()),
            ("power_transformer", FunctionTransformer(self._transform_skewed_features)),
            ("scaler", RobustScaler()),
        ]
        self._cat_steps = [
            ("imputer", SimpleImputer(strategy=self.cat_imputation)),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="error")),
        ]

        if not self.numeric_features:
            self.numeric_features = X.select_dtypes(include="number").columns.tolist()
        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(
                exclude="number"
            ).columns.tolist()

        self._steps.append(
            (
                "clean",
                ColumnTransformer(
                    [
                        ("num", Pipeline(self._num_steps), self.numeric_features),
                        ("cat", Pipeline(self._cat_steps), self.categorical_features),
                    ]
                ),
            )
        )

        if self.feature_selection:
            self._steps.append(
                (
                    "feature_selection",
                    SelectFromModel(
                        ExtraTreesClassifier(random_state=self.random_state)
                    ),
                )
            )

        if self.pca:
            self._steps.append(("pca", PCA(n_components=0.9, svd_solver="full")))

        self.pipeline_ = Pipeline(self._steps)
        self.pipeline_.fit(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series) -> (np.ndarray, np.ndarray):
        if not self.pipeline_:
            raise RuntimeError("The pipeline has not been fitted yet.")
        return self.pipeline_.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> (np.ndarray, np.ndarray):
        self.fit(X, y)
        return self.transform(X, y)

    @staticmethod
    def _transform_skewed_features(X: np.ndarray) -> np.ndarray:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        for i in range(X.shape[1]):
            feature = X[:, i]
            feature_skewness = skew(feature)
            if abs(feature_skewness) > 0.5:
                feature_reshaped = feature.reshape(-1, 1)
                transformed_feature = pt.fit_transform(feature_reshaped)
                X[:, i] = transformed_feature.flatten()
        return X
