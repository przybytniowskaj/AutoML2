import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .evaluation import ModelEvaluator
from .model_selection import ModelSelector
from .preprocessing import DataPreprocessor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Mamut:
    def __init__(self, preprocess: bool = True, **preprocessor_kwargs):
        self.preprocess = preprocess

        if self.preprocess:
            self.preprocessor = (
                DataPreprocessor(**preprocessor_kwargs) if preprocess else None
            )
        self.model_selector = None
        self.fitted_models_ = None
        self.cv_scores_ = None
        self.results_df_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        if self.preprocess:
            X_train = self.preprocessor.fit_transform(X_train, y_train)

        self.model_selector = ModelSelector(X_train, y_train)
        fitted_models, cv_scores = self.model_selector.compare_models()

        self.fitted_models_ = [
            Pipeline([("preprocessor", self.preprocessor), ("model", model)])
            for model in fitted_models
        ]
        self.cv_scores_ = cv_scores

        best_model = fitted_models[0]

        y_pred = best_model.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)

        log.info(f"Best model: {best_model.named_steps['model'].__class__.__name__}")
        log.info(f"Best score: {test_score:.4f}")

        evaluator = ModelEvaluator(fitted_models, X_test, y_test)
        self.results_df_ = evaluator.evaluate()
        evaluator.plot_results()

        models_dir = "fitted_models"
        os.makedirs(models_dir, exist_ok=True)
        for model in self.fitted_models_:
            model_name = model.named_steps["model"].__class__.__name__
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            log.info(f"Saved model {model_name} to {model_path}")

        return best_model
