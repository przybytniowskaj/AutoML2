from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mamut.preprocessing.handlers import (
    handle_categorical,
    handle_extraction,
    handle_missing_categorical,
    handle_missing_numeric,
    handle_outliers,
    handle_scaling,
    handle_selection,
    handle_skewed,
)


class Preprocessor:
    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        num_imputation: Literal[
            "iterative", "knn", "mean", "median", "constant"
        ] = "knn",
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
        self.inf_pipe_ = None
        self.imbalanced_ = None
        self.missing_ = None

        self.imbalanced_trans_ = None
        self.outlier_trans_ = None
        self.missing_num_trans_ = None
        self.missing_cat_trans_ = None
        self.cat_trans_ = None
        self.skew_trans_ = None
        self.skewed_ = None
        self.scaler_ = None
        self.sel_trans_ = None
        self.ext_trans_ = None
        self.fitted = False
        self.skewed_feature_names_ = None
        self.selected_features_ = None
        self.pca_loadings_ = None
        self.missing_numeric_ = None
        self.missing_categorical_ = None
        self.has_numeric_ = None
        self.has_categorical_ = None

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> (np.ndarray, np.ndarray, Pipeline):
        if not self.numeric_features:
            self.numeric_features = X.select_dtypes(include="number").columns.tolist()
        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(
                exclude="number"
            ).columns.tolist()

        self.has_numeric_ = len(self.numeric_features) > 0
        self.has_categorical_ = len(self.categorical_features) > 0

        if self.has_numeric_:
            if X[self.numeric_features].isnull().sum().sum() > 0:
                self.missing_ = True
                self.missing_numeric_ = True
                X, self.missing_num_trans_ = handle_missing_numeric(
                    X, self.numeric_features, self.num_imputation
                )

        if self.has_categorical_:
            if X[self.categorical_features].isnull().sum().sum() > 0:
                self.missing_ = True
                self.missing_categorical_ = True
                X, self.missing_cat_trans_ = handle_missing_categorical(
                    X, self.categorical_features, self.cat_imputation
                )

        X, y, self.outlier_trans_ = handle_outliers(
            X, y, self.numeric_features, random_state=self.random_state
        )

        X, self.cat_trans_ = handle_categorical(X, self.categorical_features)

        X, self.skew_trans_, self.skewed_feature_names_ = handle_skewed(
            X, self.numeric_features
        )
        X, self.scaler_ = handle_scaling(X, self.numeric_features)

        if self.feature_selection:
            X, self.sel_trans_, self.selected_features_ = handle_selection(
                X, y, threshold=0.05, random_state=self.random_state
            )

        if self.pca:
            X, self.ext_trans_, self.pca_loadings_ = handle_extraction(
                X, threshold=0.95, random_state=self.random_state
            )

        # TODO: Implement imbalanced handling
        # if self.imbalanced_:
        #     X, y, self.imbalanced_trans_ = handle_imbalanced(X, y)

        self.skewed_ = len(self.skewed_feature_names_) > 0
        self.fitted = True

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        return X, y

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted.")

        if self.missing_num_trans_:
            X[self.numeric_features] = self.missing_num_trans_.transform(
                X[self.numeric_features]
            )

        if self.missing_cat_trans_:
            X[self.categorical_features] = self.missing_cat_trans_.transform(
                X[self.categorical_features]
            )
            encoded_features = self.cat_trans_.transform(X[self.categorical_features])
            encoded_features_df = pd.DataFrame(
                encoded_features,
                columns=self.cat_trans_.get_feature_names_out(
                    self.categorical_features
                ),
            )
            X = X.drop(columns=self.categorical_features).join(encoded_features_df)

        if self.skewed_:
            X[self.skewed_feature_names_] = self.skew_trans_.transform(
                X[self.skewed_feature_names_]
            )

        X[self.numeric_features] = self.scaler_.transform(X[self.numeric_features])

        if self.feature_selection:
            X = self.sel_trans_.transform(X)
            X = pd.DataFrame(X, columns=self.selected_features_)

        if self.pca:
            X = self.ext_trans_.transform(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        return X
