import logging
import os
import time
from typing import List, Literal, Optional

import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from mamut.preprocessing.preprocessing import Preprocessor
from mamut.utils.utils import metric_dict

from .evaluation import ModelEvaluator  # noqa
from .model_selection import ModelSelector

# from xgboost import XGBClassifier


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
        self.fitted_models_: Optional[List[Pipeline]] = None
        self.best_model_: Optional[Pipeline] = None

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
        y = pd.Series(y)

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
        evaluator = ModelEvaluator(
            self.raw_fitted_models_,
            X_test=self.X_test,
            y_test=self.y_test,
            X_train = self.X_train,
            y_train = self.y_train,
            X=self.X,
            y=self.y,
            optimizer=self.optimization_method,
            metric=self.score_metric.__name__,
            n_trials=self.n_iterations,
            excluded_models=self.exclude_models,
            studies=self.optuna_studies_,
            training_summary=self.training_summary_,
            pca_loadings=self.preprocessor.pca_loadings_,
            binary=self.model_selector.binary,
        )

        evaluator.evaluate_to_html(self.training_summary_)
        # evaluator.plot_results()

    def _prep_preprocessing_steps_list(self) -> List[str]:
        return []

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


    def create_greedy_ensemble2(self, max_models=6):
        self._check_fitted()

        if max_models > len(self.raw_fitted_models_):
            max_models = len(self.raw_fitted_models_)
            log.info(
                f"Max models set to {max_models} as there are only {len(self.raw_fitted_models_)} models available"
                f"in the bag-of-models used in this experiment.")

        # Sort models by their performance
        sorted_models = sorted(
            self.raw_fitted_models_.items(),
            key=lambda item: self.score_metric(self.y_test, item[1].predict(self.X_test)),
            reverse=True
        )

        # Start with the best and second best models
        ensemble_models = [sorted_models[0], sorted_models[1]]
        best_score = self.score_metric(self.y_test,
                                       self._create_stacking_classifier(ensemble_models).predict(self.X_test))

        # Greedily add models to the ensemble
        for model in sorted_models[2:max_models]:
            candidate_ensemble = ensemble_models + [model]
            candidate_stacking_clf = self._create_stacking_classifier(candidate_ensemble)
            candidate_stacking_clf.fit(self.X_train, self.y_train)
            score = self.score_metric(self.y_test, candidate_stacking_clf.predict(self.X_test))

            if score > best_score:
                best_score = score
                ensemble_models.append(model)

        # Create the final stacking classifier
        final_stacking_clf = self._create_stacking_classifier(ensemble_models)
        final_stacking_clf.fit(self.X_train, self.y_train)
        self.ensemble_ = final_stacking_clf

        log.info(f"Created greedy ensemble with {len(ensemble_models)} models. Best score: {best_score:.4f}")
        return self.ensemble_


    def _create_stacking_classifier(self, models):
        estimators = [(name, clone(model)) for name, model in models]
        return StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())


    def _calculate_disagreement(self, model1, model2, X_test):
        """Calculate disagreement between two models' predictions."""
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        return np.mean(pred1 != pred2)

    def _ensemble_selection(self, max_ensemble_size=5, voting: Literal["soft", "hard"] = "soft"):
        """Greedy algorithm to select the best subset of models for ensemble."""
        #  TODO: THIS IS WORK IN PROGRESS... DO NOT USE
        models = [(model.__class__.__name__, model) for model in self.raw_fitted_models_.values()]
        print(models)
        selected_models = []
        remaining_models = models.copy()
        best_score = 0
        ensemble_performance = []

        # Initialize with the best performing model on test set
        scores = {name: self.score_metric(self.y_test, model.predict(self.X_test)) for name, model in models}
        print("Scores:", scores)
        best_model_name, best_model = max(scores.items(), key=lambda item: item[1])
        print("Best model:", best_model_name, "with score:", best_model)
        selected_models.append((best_model_name, best_model))
        remaining_models.remove((best_model_name, best_model))
        best_score = scores[best_model_name]
        ensemble_performance.append(best_score)

        print(f"Starting with best model: {best_model_name} with score: {best_score}")

        # Greedily add models based on performance and diversity
        while len(selected_models) < max_ensemble_size and remaining_models:
            best_model_to_add = None
            best_new_score = best_score
            for name, model in remaining_models:
                # Test current ensemble with this model added
                current_ensemble = VotingClassifier(estimators=selected_models + [(name, model)], voting=voting)
                current_ensemble.fit(self.X_train, self.y_train)
                ensemble_score = self.score_metric(self.y_test, current_ensemble.predict(self.X_test))

                # Compute diversity with selected models
                diversity = np.mean([self._calculate_disagreement(model, selected_model[1], self.X_test)
                                     for selected_model in selected_models])

                # Score considering both accuracy improvement and diversity
                weighted_score = ensemble_score + 0.1 * diversity  # 0.1 is a diversity weight factor
                if weighted_score > best_new_score:
                    best_model_to_add = (name, model)
                    best_new_score = weighted_score

            if best_model_to_add:
                selected_models.append(best_model_to_add)
                remaining_models.remove(best_model_to_add)
                best_score = best_new_score
                ensemble_performance.append(best_new_score)
                print(f"Added {best_model_to_add[0]} to ensemble, new weighted score: {best_new_score}")
            else:
                break  # No improvement

        # Final ensemble
        final_ensemble = VotingClassifier(estimators=selected_models, voting=voting)
        final_ensemble.fit(self.X_train, self.y_train)
        return final_ensemble, ensemble_performance



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
