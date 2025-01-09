from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        num_imputation: Literal["mean", "median", "constant"] = "mean",
        cat_imputation: Literal["most_frequent", "constant"] = "most_frequent",
        scaling_method: Literal["standard", "minmax", None] = None,
        test_size: Optional[float] = 0.2,
        correlation_threshold: float = 0.98,
        feature_selection: bool = True,
        feature_selection_treshold: float = 0.8,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the DataPreprocessor.

        Parameters:
        - num_strategy: Strategy for imputing numerical values ('mean', 'median', 'constant').
        - cat_strategy: Strategy for imputing categorical values ('most_frequent', 'constant').
        - scaling_method: Scaling method for numerical features ('standard', 'minmax').
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Random state for reproducibility.
        """
        self.validate_inputs(num_imputation, cat_imputation, scaling_method)
        self.num_imputation = num_imputation
        self.cat_imputation = cat_imputation
        self.scaling_method = scaling_method
        self.test_size = test_size
        self.num_features = numeric_features
        self.cat_features = categorical_features
        self.correlation_threshold = correlation_threshold
        self.feature_selection = feature_selection
        self.feature_selection_treshold = feature_selection_treshold
        self.random_state = random_state
        self.fitted_pipeline = None

    @staticmethod
    def validate_inputs(
        num_imputation: Literal["mean", "median", "constant"],
        cat_imputation: Literal["most_frequent", "constant"],
        scaling_method: Literal["standard", "minmax", None],
    ) -> None:
        """
        Validate the input parameters.

        Parameters:
        - num_strategy: Strategy for imputing numerical values.
        - cat_strategy: Strategy for imputing categorical values.
        - scaling_method: Scaling method for numerical features.
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Random state for reproducibility.

        Raises:
        - ValueError: If any parameter is invalid.
        """
        valid_num_strategies = ["mean", "median", "constant"]
        valid_cat_strategies = ["most_frequent", "constant"]
        valid_scaling_methods = ["standard", "minmax", None]

        if num_imputation not in valid_num_strategies:
            raise ValueError(
                f"num_strategy must be one of {valid_num_strategies}, got '{num_imputation}'."
            )

        if cat_imputation not in valid_cat_strategies:
            raise ValueError(
                f"cat_strategy must be one of {valid_cat_strategies}, got '{cat_imputation}'."
            )

        if scaling_method not in valid_scaling_methods:
            raise ValueError(
                f"scaling_method must be one of {valid_scaling_methods}, got '{scaling_method}'."
            )

    @staticmethod
    def detect_outliers(series: pd.Series) -> bool:
        """
        Detect outliers using the IQR method.

        Parameters:
        - series: Pandas Series to check for outliers.

        Returns:
        - Boolean indicating whether outliers are detected.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        return outliers.any()

    @staticmethod
    def is_gaussian(series: pd.Series, alpha: float = 0.05) -> bool:
        """
        Check if a feature follows a Gaussian distribution using the Shapiro-Wilk test.

        Parameters:
        - series: Pandas Series to test.
        - alpha: Significance level.

        Returns:
        - Boolean indicating whether the feature is Gaussian.
        """
        stat, p = shapiro(series)
        return p > alpha

    def choose_scaler(self, X: pd.DataFrame) -> dict:
        """
        Choose scalers dynamically based on outlier detection and Gaussian distribution
        or use the predefined scaling_method if it is set.

        Parameters:
        - X: Input DataFrame.
        - num_features: List of numerical feature names.

        Returns:
        - Dictionary mapping features to scalers.
        """
        scalers = {}
        if self.scaling_method:
            scaler = (
                StandardScaler()
                if self.scaling_method == "standard"
                else MinMaxScaler()
            )
            scalers = {feature: scaler for feature in self.num_features}
        else:
            for feature in self.num_features:
                series = X[feature]
                has_outliers = self.detect_outliers(series)
                is_gaussian = self.is_gaussian(series)
                scalers[feature] = (
                    StandardScaler() if has_outliers or is_gaussian else MinMaxScaler()
                )
        return scalers

    def create_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Create a preprocessing pipeline with dynamic scaling for numerical features.

        Parameters:
        - X: Input DataFrame.
        - num_features: List of numerical feature names.
        - cat_features: List of categorical feature names.

        Returns:
        - A sklearn Pipeline object.
        """
        scalers = self.choose_scaler(X)
        num_transformer_steps = [
            ("imputer", SimpleImputer(strategy=self.num_imputation))
        ]
        for feature in self.num_features:
            num_transformer_steps.append((f"scaler_{feature}", scalers[feature]))

        num_transformer = Pipeline(num_transformer_steps)
        cat_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=self.cat_imputation)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("num", num_transformer, self.num_features),
                ("cat", cat_transformer, self.cat_features),
            ]
        )
        return preprocessor

    def remove_highly_correlated_features(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove one column from each pair of highly correlated features.

        Parameters:
        - X: Input DataFrame with numerical features.
        - threshold: Correlation threshold above which features are considered highly correlated.

        Returns:
        - DataFrame with highly correlated features removed.
        """
        X_copy = X[self.num_features].copy()
        corr_matrix = X_copy.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]
        return X.drop(columns=to_drop), [
            feature for feature in self.num_features if feature not in to_drop
        ]

    def select_important_features(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the top `threshold_percent` most important features using a RandomForestClassifier
        after the data has been transformed.

        Parameters:
        - X_train_transformed: Transformed training data.
        - X_test_transformed: Transformed test data.
        - y_train: Target variable for training.
        - threshold_percent: Proportion of features to retain (default is 80%).

        Returns:
        - X_train_reduced: Training data with selected features.
        - X_test_reduced: Test data with selected features.
        - selected_features: List of selected feature names.
        """
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(X_train, y_train.values.ravel())

        importances = rf.feature_importances_
        num_features_to_select = int(len(importances) * self.feature_selection_treshold)

        selector = SelectFromModel(
            rf, max_features=num_features_to_select, threshold=-np.inf
        )
        selector.fit(X_train, y_train.values.ravel())

        selected_columns = selector.get_support(indices=True)
        X_train = X_train[:, selected_columns]
        X_test = X_test[:, selected_columns]

        # selected_feature_names = [f"feature_{i}" for i in selected_columns]
        return X_train, X_test

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
        """
        Split, fit, and transform the data.

        Parameters:
        - X: Input DataFrame.
        - y: Target Series or DataFrame.

        Returns:
        - X_train, X_test, y_train, y_test: Transformed and split data.
        """
        if not self.num_features:
            self.num_features = X.select_dtypes(include=["number"]).columns
        if not self.cat_features:
            self.cat_features = X.select_dtypes(exclude=["number"]).columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train, self.num_features = self.remove_highly_correlated_features(X_train)
        self.fitted_pipeline = self.create_pipeline(X_train)
        X_train = self.fitted_pipeline.fit_transform(X_train)
        X_test = self.fitted_pipeline.transform(X_test)

        if self.feature_selection:
            X_train, X_test = self.select_important_features(X_train, X_test, y_train)

        return X_train, y_train, X_test, y_test

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data (after fitting).

        Parameters:
        - X: Input DataFrame.

        Returns:
        - Transformed data.
        """
        if self.fitted_pipeline is None:
            raise ValueError(
                "Pipeline has not been fitted. Call 'fit_transform' first."
            )

        return self.fitted_pipeline.transform(X)
