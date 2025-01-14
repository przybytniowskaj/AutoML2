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
