import logging
import os
import time
from typing import List, Literal, Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier

from mamut.preprocessing.preprocessing import Preprocessor

from .evaluation import ModelEvaluator  # noqa
from .model_selection import ModelSelector
from mamut.utils.utils import metric_dict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class Mamut:
    def __init__(
        self,
        preprocess: bool = True,
        imb_threshold: float = 0.10,
        exclude_models: Optional[List[str]] = None,
        score_metric: Literal[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "jaccard",
            "roc_auc",
        ] = "f1",
        optimization_method: Literal["random_search", "bayes"] = "bayes",
        n_iterations: Optional[int] = 50,
        random_state: Optional[int] = 42,
        **preprocessor_kwargs,
    ):
        self.preprocess = preprocess
        self.imb_threshold = imb_threshold
        self.exclude_models = exclude_models
        self.score_metric = metric_dict[score_metric]
        self.optimization_method = optimization_method
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.preprocessor = Preprocessor(**preprocessor_kwargs) if preprocess else None
        self.le = LabelEncoder()
        self.model_selector = None

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # self.raw_models_: Optional[List[Union[BaseEstimator, XGBClassifier]]] = None
        self.raw_fitted_models_ = None
        self.fitted_models_ : Optional[List[Pipeline]] = None
        self.best_model_ : Optional[Pipeline] = None
        self.best_score_ = None
        self.training_summary_ = None
        self.optuna_studies_ = None

        self.ensemble_: Optional[Pipeline] = None
        self.greedy_ensemble_: Optional[Pipeline] = None
        self.ensemble_models_ = None
        self.imbalanced_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Mamut._check_categorical(y)
        if y.value_counts(normalize=True).min() < self.imb_threshold:
            self.imbalanced_ = True

        y = self.le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        if self.preprocess:
            X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)
            X_test = self.preprocessor.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.y = y

        self.model_selector = ModelSelector(
            X_train,
            y_train,
            X_test,
            y_test,
            exclude_models=self.exclude_models,
            score_metric=self.score_metric,
            optimization_method=self.optimization_method,
            n_iterations=self.n_iterations,
            random_state=self.random_state,
        )

        (
            best_model,
            params_for_best_model,
            score_for_best_model,
            fitted_models,
            training_summary,
            studies,
        ) = self.model_selector.compare_models()

        self.raw_fitted_models_ = fitted_models
        self.optuna_studies_ = studies
        self.fitted_models_ = [
            Pipeline([("preprocessor", self.preprocessor), ("model", model)])
            for model in fitted_models.values()
        ]

        self.best_score_ = score_for_best_model
        # TODO: This works???
        self.best_model_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", best_model)]
        )
        self.training_summary_ = training_summary

        log.info(f"Best model: {best_model.__class__.__name__}")

        # Models_dir with time signature
        cwd = os.getcwd()
        models_dir = os.path.join(
            cwd,
            "fitted_models",
            str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())),
        )
        os.makedirs(models_dir, exist_ok=True)
        for model in self.fitted_models_:
            model_name = model.named_steps["model"].__class__.__name__
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            log.info(f"Saved model {model_name} to {model_path}")

        return self.best_model_

    def predict(self, X: pd.DataFrame):
        return self._predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self._predict(X, proba=True)

    def evaluate(self) -> None:
        self._check_fitted()

        # TODO: CHANGE, only for debug
        m = KNeighborsClassifier(n_neighbors=5)
        m.fit(self.X_train, self.y_train)
        evaluator = ModelEvaluator(self.raw_fitted_models_, X_test=self.X_test, y_test=self.y_test,
                                   X=self.X, y=self.y, optimizer=self.optimization_method,
                                   metric=self.score_metric.__name__, n_trials=self.n_iterations,
                                   excluded_models=self.exclude_models, studies=self.optuna_studies_,
                                   )

        evaluator.evaluate_to_html(self.training_summary_)
        # evaluator.plot_results()

    def save_best_model(self, path: str) -> None:
        # TODO: Think if necessary (all models are saved in the fitted_models dir)
        self._check_fitted()
        save_path = os.path.join(
            path, f"{self.best_model_.named_steps['model'].__class__.__name__}.joblib"
        )
        joblib.dump(self.best_model_, save_path)
        log.info(f"Saved best model to {save_path}")

    def create_ensemble(self, voting: Literal["soft", "hard"] = "soft") -> Pipeline:
        self._check_fitted()

        ensemble = VotingClassifier(
            estimators=[
                (
                    model.named_steps["model"].__class__.__name__,
                    clone(model.named_steps["model"]),
                )
                for model in self.fitted_models_
            ],
            voting=voting,
        )

        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        score = self.score_metric(self.y_test, y_pred)

        self.ensemble_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", ensemble)]
        )
        log.info(
            f"Created ensemble with all models and voting='{voting}'. "
            f"Ensemble score on test set: {score:.4f} {self.score_metric.__name__}"
        )

        return self.ensemble_

    def create_greedy_ensemble(
        self, n_models: int = 6, voting: Literal["soft", "hard"] = "soft"
    ) -> Pipeline:
        self._check_fitted()

        # Initialize the ensemble with the best model
        ensemble_models = [self.best_model_.named_steps["model"]]
        ensemble_scores = [self.best_score_]

        for _ in range(n_models - 1):
            best_score = 0
            best_model = None

            for model in self.fitted_models_:
                candidate_ensemble = ensemble_models + [model.named_steps["model"]]
                candidate_voting_clf = VotingClassifier(
                    estimators=[
                        (f"model_{i}", clone(m))
                        for i, m in enumerate(candidate_ensemble)
                    ],
                    voting=voting,
                )
                candidate_voting_clf.fit(self.X_train, self.y_train)
                score = self.score_metric(
                    self.y_test, candidate_voting_clf.predict(self.X_test)
                )

                if score > best_score:
                    best_score = score
                    best_model = model.named_steps["model"]

            ensemble_models.append(best_model)
            ensemble_scores.append(best_score)

        ensemble = VotingClassifier(
            estimators=[
                (f"model_{i}", clone(m)) for i, m in enumerate(ensemble_models)
            ],
            voting=voting,
        )
        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        score = self.score_metric(self.y_test, y_pred)

        self.ensemble_models_ = ensemble_models
        self.greedy_ensemble_ = Pipeline(
            [("preprocessor", self.preprocessor), ("model", ensemble)]
        )

        log.info(
            f"Created greedy ensemble with voting='{voting}' \n"
            f"and {n_models} models: {[m.__class__.__name__ for m in ensemble_models]} \n"
            f"Ensemble score on test set: {score:.4f} {self.score_metric.__name__}"
        )

        return self.greedy_ensemble_

    def _predict(self, X: pd.DataFrame, proba: bool = False):
        self._check_fitted()
        if proba:
            return self.best_model_.predict_proba(X)
        return self.best_model_.predict(X)

    def _check_fitted(self):
        if not self.best_model_:
            raise RuntimeError(
                "Can't predict because no model has been fitted. "
                "Please call fit() method first."
            )

    @staticmethod
    def _check_categorical(y):
        if pd.api.types.is_float_dtype(y):
            raise ValueError("Target variable must be categorical.")
