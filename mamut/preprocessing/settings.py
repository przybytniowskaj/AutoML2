from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

imputer_mapping = {
    "iterative": IterativeImputer,
    "knn": KNNImputer,
    "mean": lambda: SimpleImputer(strategy="mean"),
    "median": lambda: SimpleImputer(strategy="median"),
    "constant": lambda: SimpleImputer(strategy="constant"),
}

scaler_mapping = {
    "standard": StandardScaler,
    "robust": RobustScaler,
}

resampler_mapping = {
    "SMOTE": SMOTE,
    "undersample": RandomUnderSampler,
    "combine": SMOTETomek,
}
