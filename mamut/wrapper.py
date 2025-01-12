import logging
import os
import time
from typing import List, Literal, Optional
import joblib

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .evaluation import ModelEvaluator
from .model_selection import ModelSelector
from .preprocessing import DataPreprocessor
from .utils import metric_dict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Mamut:
    def __init__(self, preprocess: bool = True,
                 exclude_models: Optional[List[str]] = None,
                 score_metric: Literal["accuracy", "precision", "recall", "f1",
                 "balanced_accuracy", "jaccard", "roc_auc"] = "roc_auc",
                 optimization_method: Literal["random_search", "bayes"] = "bayes",
                 n_iterations: Optional[int] = 50,
                 random_state: Optional[int] = 42,
                 **preprocessor_kwargs):


        self.preprocess = preprocess
        self.exclude_models = exclude_models
        self.score_metric = metric_dict[score_metric]
        self.optimization_method = optimization_method
        self.n_iterations = n_iterations
        self.random_state = random_state

        if self.preprocess:
            self.preprocessor = (
                DataPreprocessor(**preprocessor_kwargs) if preprocess else None
            )

        self.model_selector = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ensemble = None
        self.ensemble_models = None

        self.fitted_models_ = None
        self.best_model_ = None
        self.best_score_ = None
        self.results_df_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )


        if self.preprocess:
            X_train = self.preprocessor.fit_transform(X_train, y_train)

        # TODO: Clean X_test and y_test with .transform() ?
        if self.preprocess:
            X_test = self.preprocessor.transform(X_test, y_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model_selector = ModelSelector(X_train,
                                            y_train,
                                            X_test,
                                            y_test,
                                            exclude_models=self.exclude_models,
                                            score_metric=self.score_metric,
                                            optimization_method=self.optimization_method,
                                            n_iterations=self.n_iterations,
                                            random_state=self.random_state
                                            )

        best_model, params_for_best_model, score_for_best_model, fitted_models, training_report = self.model_selector.compare_models()

        # self.fitted_models_ = [
        #     Pipeline([("preprocessor", self.preprocessor), ("model", model)])
        #     for model in fitted_models
        # ]
        # TODO: Which one?
        self.fitted_models_ = fitted_models

        self.best_score_ = score_for_best_model
        self.best_model_ = best_model

        y_pred = best_model.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)

        log.info(f"Best model: {best_model.named_steps['model'].__class__.__name__}")
        log.info(f"Best score: {test_score:.4f}")

        evaluator = ModelEvaluator(fitted_models, X_test, y_test)
        self.results_df_ = evaluator.evaluate()
        evaluator.plot_results()

        # Models_dir with time signature
        models_dir = "fitted_models" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(models_dir, exist_ok=True)
        for model in self.fitted_models_:
            model_name = model.named_steps["model"].__class__.__name__
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            log.info(f"Saved model {model_name} to {model_path}")

        return best_model

    def create_ensemble(self, voting: Literal["soft", "hard"] = "soft") -> VotingClassifier:
        if not self.fitted_models_:
            raise RuntimeError("Can't create ensemble because no models have been fitted. "
                               "Please call fit() method first.")


        ensemble = VotingClassifier(estimators=[
            (model.__class__.__name__, clone(model))
            for model in self.fitted_models_
        ], voting=voting) # TODO: Change if fitted_models_ is a list of pipelines!
        # ensemble = VotingClassifier(estimators=[(model.named_steps["model"].__class__.__name__, model.named_steps["model"]) for model in self.fitted_models_], voting=voting)

        # if self.preprocess:
        #     X_train = self.preprocessor.transform(self.X_train)

        ensemble.fit(self.X_train, self.y_train)
        self.ensemble = ensemble
        log.info(f"Created ensemble with voting='{voting}'")

        return ensemble


    def create_greedy_ensemble(self, n_models: int = 6, voting: Literal["soft", "hard"] = "soft") -> VotingClassifier:
        if not self.fitted_models_:
            raise RuntimeError("Can't create ensemble because no models have been fitted. "
                               "Please call fit() method first.")

        # Initialize the ensemble with the best model
        ensemble_models = [self.best_model_]
        ensemble_scores = [self.best_score_]

        for _ in range(n_models - 1):
            best_score = 0
            best_model = None

            for model in self.fitted_models_:
                candidate_ensemble = ensemble_models + [model]
                candidate_voting_clf = VotingClassifier(
                    estimators=[(f"model_{i}", clone(m)) for i, m in enumerate(candidate_ensemble)],
                    voting=voting
                )
                candidate_voting_clf.fit(self.X_train, self.y_train)
                score = self.score_metric(self.y_test, candidate_voting_clf.predict(self.X_test))

                if score > best_score:
                    best_score = score
                    best_model = model

            ensemble_models.append(best_model)
            ensemble_scores.append(best_score)

        self.ensemble = VotingClassifier(
            estimators=[(f"model_{i}", clone(m)) for i, m in enumerate(ensemble_models)],
            voting=voting
        )
        self.ensemble_models = ensemble_models

        self.ensemble.fit(self.X_train, self.y_train)
        log.info(
            f"Created greedy ensemble with voting='{voting}' and models: {[m.__class__.__name__ for m in ensemble_models]}")

        return self.ensemble

