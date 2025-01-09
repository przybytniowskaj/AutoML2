import warnings

import catboost as cb
import lightgbm as lgb
import optuna
import xgboost as xgb
from package_name.utils import model_param_dict, sample_parameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


class ModelSelector:
    def __init__(self, X_train, y_train, X_test, y_test, models=None):
        self.X_train = X_train
        self.y_train = y_train.values.ravel()
        self.X_test = X_test
        self.y_test = y_test.values.ravel()
        self.models = (
            models
            if models
            else [
                LogisticRegression(max_iter=1000),
                RandomForestClassifier(),
                SVC(),
                xgb.XGBClassifier(eval_metric="mlogloss"),
                lgb.LGBMClassifier(verbose=-1),
                cb.CatBoostClassifier(verbose=0),
                MLPClassifier(max_iter=1000),
            ]
        )

    def objective(self, trial, model):
        if model.__class__.__name__ in model_param_dict:
            param_grid = model_param_dict[model.__class__.__name__]
            if model.__class__.__name__ == "CatBoostClassifier":
                model = cb.CatBoostClassifier(verbose=0)
        else:
            raise ValueError(f"Model {model.__class__.__name__} not supported")

        param = {
            param_name: sample_parameter(trial, param_name, value)
            for param_name, value in param_grid.items()
        }

        if isinstance(model, LogisticRegression):
            if param["solver"] == "saga":
                param["penalty"] = "elasticnet"
            else:
                param["penalty"] = "l2"
                param["l1_ratio"] = None

        model.set_params(**param)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def optimize_model(self, model):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, model),
            n_trials=50,
            show_progress_bar=True,
        )
        return study.best_params, study.best_value

    def compare_models(self):
        best_score = 0
        best_model = None
        best_params = None
        fitted_models = []

        for model in self.models:
            print(f"Optimizing model: {model.__class__.__name__}")
            params, score = self.optimize_model(model)
            print(f"Best parameters: {params}, score: {score:.4f}\n")

            model.set_params(**params)
            model.fit(self.X_train, self.y_train)
            fitted_models.append(model)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

        print(
            f"Best model: {best_model.__class__.__name__} with parameters {best_params} and score {best_score:.4f}"
        )
        return best_model, best_params, best_score, fitted_models
