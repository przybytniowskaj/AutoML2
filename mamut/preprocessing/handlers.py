from typing import List, Literal

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTETomek
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

import mamut.preprocessing.settings as settings


def handle_outliers(X, y, feature_names, contamination=0.01, random_state=42):
    """
    Handles outliers in the dataset using IsolationForest.

    Parameters:
        X: ndarray or DataFrame
            Feature matrix.
        y: ndarray or Series
            Target array.
        contamination: float
            The proportion of outliers in the data.
        random_state: int
            Seed for reproducibility.

    Returns:
        X_filtered: ndarray or DataFrame
            Feature matrix with outliers removed.
        y_filtered: ndarray or Series
            Target array with outliers removed.
        transformer: IsolationForest
            Fitted IsolationForest model.
    """
    X, y = X.copy(), y.copy()
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = iso_forest.fit_predict(X[feature_names])
    mask = outliers == 1

    return X[mask], y[mask], iso_forest


# def handle_imbalanced(X, y, method='SMOTE', random_state=42):
#     """
#     Balances an imbalanced dataset using techniques from imbalanced-learn.
#
#     Parameters:
#         X: ndarray or DataFrame
#             Feature matrix.
#         y: ndarray or Series
#             Target array.
#         method: str
#             Resampling method to use. Options:
#                 - 'SMOTE': Synthetic Minority Oversampling Technique.
#                 - 'undersample': Random undersampling of majority class.
#                 - 'combine': SMOTE with Tomek links.
#         random_state: int
#             Seed for reproducibility.
#
#     Returns:
#         X_resampled: ndarray or DataFrame
#             Feature matrix after resampling.
#         y_resampled: ndarray or Series
#             Target array after resampling.
#         transformer: Object
#             Fitted resampling method instance.
#     """
#     if method == 'SMOTE':
#         resampler = SMOTE(random_state=random_state)
#     elif method == 'undersample':
#         resampler = RandomUnderSampler(random_state=random_state)
#     elif method == 'combine':
#         resampler = SMOTETomek(random_state=random_state)
#     else:
#         raise ValueError("Invalid method. Choose from 'SMOTE', 'undersample', or 'combine'.")
#
#     X_resampled, y_resampled = resampler.fit_resample(X, y)
#
#     return X_resampled, y_resampled, resampler


def handle_skewed(X: pd.DataFrame, feature_names: List[str]):
    X = X.copy()
    skewed_feature_names = []
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    for feature in feature_names:
        feature_skewness = skew(X[feature])
        if abs(feature_skewness) > 2:
            skewed_feature_names.append(feature)

    if len(skewed_feature_names) > 0:
        X[skewed_feature_names] = pt.fit_transform(X[skewed_feature_names])

    return X, pt, skewed_feature_names


def handle_missing_numeric(X, feature_names, strategy):
    if strategy not in settings.imputer_mapping.keys():
        raise ValueError(
            f"Invalid imputation strategy, choose from {settings.imputer_mapping.keys()}."
        )

    X = X.copy()
    imputer = settings.imputer_mapping[strategy]()
    imputer.fit(X[feature_names])
    X[feature_names] = imputer.transform(X[feature_names])

    return X, imputer


def handle_missing_categorical(X, feature_names, strategy: str):
    X = X.copy()
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X[feature_names])
    X[feature_names] = imputer.transform(X[feature_names])

    return X, imputer


def handle_categorical(X, feature_names):
    X = X.copy()
    encoder = OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False)
    encoder.fit(X[feature_names])
    encoded_features = encoder.transform(X[feature_names])
    encoded_features_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(feature_names),
        index=X.index,
    )
    X = X.drop(columns=feature_names).join(encoded_features_df)

    return X, encoder


def handle_scaling(
    X: pd.DataFrame, feature_names: List[str], strategy: Literal["standard", "robust"]
):
    if strategy not in ["standard", "robust"]:
        raise ValueError(
            f"Invalid scaling strategy, choose from {settings.scaler_mapping.keys()}."
        )

    X = X.copy()
    scaler = settings.scaler_mapping[strategy]()
    scaler.fit(X[feature_names])
    X[feature_names] = scaler.transform(X[feature_names])

    return X, scaler


def handle_selection(X, y, threshold=0.05, random_state=42):
    X = X.copy()
    selector = SelectFromModel(
        ExtraTreesClassifier(random_state=random_state), threshold=threshold
    )
    selector.fit(X, y)
    X_selected = selector.transform(X)
    selected_features = X.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    return X_selected_df, selector, selected_features


def handle_extraction(X, threshold=0.95, random_state=42) -> (np.ndarray, PCA):
    X = X.copy()
    extractor = PCA(
        n_components=threshold, svd_solver="full", random_state=random_state
    )
    extractor.fit(X)
    X_extracted = extractor.transform(X)
    loadings = extractor.components_

    return X_extracted, extractor, loadings
