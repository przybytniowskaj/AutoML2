import time
import warnings
from typing import List, Optional, Literal

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler, RandomSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from mamut.utils import model_param_dict, sample_parameter, metric_dict, adjust_search_spaces

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


class ModelSelector:
    def __init__(self, X_train,
                 y_train,
                 X_test,
                 y_test,
                 exclude_models: Optional[List[str]] = None,
                 score_metric: Literal["accuracy", "precision", "recall", "f1",
                 "balanced_accuracy", "jaccard", "roc_auc"] = "roc_auc",
                 optimization_method: Literal["random_search", "bayes"] = "bayes",
                 n_iterations: Optional[int] = 50,
                 random_state: Optional[int] = 42):

        self.X_train = X_train
        self.y_train = y_train.values.ravel()
        self.X_test = X_test
        self.y_test = y_test.values.ravel()
        self.models = (
            # Include models that are in model_param_dict but not those in exclude_models
            [eval(model)(random_state=random_state) if "random_state" in eval(model)().get_params() else eval(model)()
             for model in model_param_dict.keys()
             if model.__name__ not in exclude_models
             ]
        )
        self.score_metric = metric_dict[score_metric]
        self.optuna_sampler = TPESampler(seed=random_state) if optimization_method == "bayes" else RandomSampler(seed=random_state)
        self.n_iterations = n_iterations
        self.SKF_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # TODO: Jeśli jest multi-class, to dostosowanie LR i innych

    def objective(self, trial, model):
        if model.__class__.__name__ in model_param_dict:
            param_grid = model_param_dict[model.__class__.__name__]
        else:
            raise ValueError(f"Model {model.__class__.__name__} not supported")

        param = {
            param_name: sample_parameter(trial, param_name, value)
            for param_name, value in param_grid.items()
        }

        # Adjust if needed for compatibility between hyperparameters
        param = adjust_search_spaces(param, model)

        model.set_params(**param)

        cv_scores = []
        for train_idx, val_idx in self.SKF_.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            # Reinitialize the model
            model = model.__class__(**model.get_params())
            model.fit(X_train_fold, y_train_fold)
            val_pred = model.predict(X_val_fold)

            cv_scores.append(self.score_metric(self.y_train[val_idx], val_pred))
        mean_cv_score = np.mean(cv_scores)

        # Optimize wrt. the chosen metric
        return mean_cv_score

    def optimize_model(self, model):
        study = optuna.create_study(direction="maximize", sampler=self.optuna_sampler)
        start_time = time.time()
        study.optimize(
            lambda trial: self.objective(trial, model),
            n_trials=self.n_iterations,
            show_progress_bar=True,
        )
        end_time = time.time()
        duration = end_time - start_time
        return study.best_params, study.best_value, duration

    def compare_models(self):
        best_model = None
        score_for_best_model = 0
        params_for_best_model = None
        fitted_models = []
        training_report = pd.DataFrame()
        scores_on_test = {}

        for model in self.models:
            print(f"Optimizing model: {model.__class__.__name__}")
            params, score, duration = self.optimize_model(model)
            print(f"Best parameters: {params}, score: {score:.4f} {self.score_metric.__name__}\n")

            # Reinitialize the model with the best parameters
            model.set_params(**params)
            model = model.__class__(**model.get_params())

            model.fit(self.X_train, self.y_train)
            fitted_models.append(model)

            score_on_test = self.score_metric(self.y_test, model.predict(self.X_test))

            if score_on_test > score_for_best_model:
                score_for_best_model = score_on_test
                best_model = model
                params_for_best_model = params

            # Save the training report
            scores_on_test = self.score_model_with_metrics(model)
            training_report = training_report.append(
                {
                    "model": model.__class__.__name__,
                    **scores_on_test,
                    "duration": duration,
                },
                ignore_index=True
            )

        print(
            f"Found best model: {best_model.__class__.__name__} with parameters {params_for_best_model} and score {score_for_best_model:.4f}"
        )
        return best_model, params_for_best_model, score_for_best_model, fitted_models, training_report


    def score_model_with_metrics(self, fitted_model):
        # Check if the model is fitted
        if not hasattr(fitted_model, "predict"):
            raise ValueError("The model is not fitted and can not be scored with any metric.")

        y_pred = fitted_model.predict(self.X_test)
        results = {
            "accuracy_score": accuracy_score(self.y_test, y_pred),
            "balanced_accuracy_score": balanced_accuracy_score(self.y_test, y_pred),
            "precision_score": precision_score(self.y_test, y_pred, average="weighted"),
            "recall_score": recall_score(self.y_test, y_pred, average="weighted"),
            "f1_score": f1_score(self.y_test, y_pred, average="weighted"),
            "jaccard_score": jaccard_score(self.y_test, y_pred, average="weighted"),
            "roc_auc_score": roc_auc_score(self.y_test, y_pred, multi_class="ovr"), # TODO: Will not work for multi-class
        }

        # Change the order of columns so that the self.score_metric is first
        results = {self.score_metric.__name__: results.pop(self.score_metric.__name__), **results}
        return results
