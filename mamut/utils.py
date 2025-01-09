lr_params = {
    "C": (1e-4, 1e4, "log"),
    "l1_ratio": (1e-4, 1.0, "log"),
    "class_weight": (["balanced"], "categorical"),
    "max_iter": (1500, 1500, "int"),
    "solver": (["saga", "lbfgs", "liblinear"], "categorical"),
}

tree_params = {
    "n_estimators": (10, 1000, "int"),
    "criterion": (["gini", "entropy", "log_loss"], "categorical"),
    "bootstrap": ([True], "categorical"),
    "max_samples": (0.5, 1, "float"),
    "max_features": (0.1, 0.9, "float"),
    "min_samples_leaf": (0.05, 0.25, "float"),
}

xgb_params = {
    "n_estimators": (10, 2000, "int"),
    "learning_rate": (1e-4, 0.4, "log"),
    "subsample": (0.25, 1.0, "float"),
    "booster": (["gbtree"], "categorical"),
    "max_depth": (1, 15, "int"),
    "min_child_weight": (1, 128, "float"),
    "colsample_bytree": (0.2, 1.0, "float"),
    "colsample_bylevel": (0.2, 1.0, "float"),
    "reg_alpha": (1e-4, 512.0, "log"),
    "reg_lambda": (1e-3, 1e3, "log"),
}

svc_params = {
    "C": (1e-4, 1e4, "log"),
    "kernel": (["linear", "poly", "rbf", "sigmoid"], "categorical"),
    "gamma": (1e-4, 1.0, "log"),
    "class_weight": (["balanced"], "categorical"),
}

lgb_params = {
    "num_leaves": (15, 255, "int"),
    "learning_rate": (1e-4, 0.4, "log"),
    "n_estimators": (10, 2000, "int"),
    "max_depth": (-1, 15, "int"),
    "min_child_samples": (5, 100, "int"),
    "subsample": (0.4, 1.0, "float"),
    "colsample_bytree": (0.4, 1.0, "float"),
    "reg_alpha": (1e-4, 10.0, "log"),
    "reg_lambda": (1e-4, 10.0, "log"),
}

cb_params = {
    "iterations": (10, 2000, "int"),
    "depth": (3, 10, "int"),
    "learning_rate": (1e-4, 0.4, "log"),
    "l2_leaf_reg": (1.0, 10.0, "log"),
    "border_count": (32, 255, "int"),
    "subsample": (0.5, 1.0, "float"),
    "colsample_bylevel": (0.4, 1.0, "float"),
}

mlp_params = {
    "hidden_layer_sizes": ([(50,), (100,), (200,)], "categorical"),
    "activation": (["identity", "logistic", "tanh", "relu"], "categorical"),
    "solver": (["lbfgs", "sgd", "adam"], "categorical"),
    "alpha": (1e-5, 1e-2, "log"),
    "learning_rate": (["constant", "invscaling", "adaptive"], "categorical"),
    "learning_rate_init": (1e-4, 1e-1, "log"),
    "power_t": (0.1, 0.9, "float"),
    "max_iter": (200, 2000, "int"),
    "momentum": (0.5, 0.9, "float"),
}

model_param_dict = {
    "LogisticRegression": lr_params,
    "RandomForestClassifier": tree_params,
    "SVC": svc_params,
    "XGBClassifier": xgb_params,
    "LGBMClassifier": lgb_params,
    "CatBoostClassifier": cb_params,
    "MLPClassifier": mlp_params,
}


def sample_parameter(trial, param_name, value):
    """Sample a parameter value based on its distribution type."""
    if len(value) == 3:
        low, high, dist_type = value
        if dist_type == "log":
            return trial.suggest_float(param_name, low, high, log=True)
        elif dist_type == "float":
            return trial.suggest_float(param_name, low, high)
        else:
            return trial.suggest_int(param_name, low, high)
    elif len(value) == 2:
        options, dist_type = value
        return trial.suggest_categorical(param_name, options)
    else:
        raise ValueError("Invalid parameter value")
