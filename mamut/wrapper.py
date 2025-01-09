import logging

from sklearn.model_selection import train_test_split

from .evaluation import ModelEvaluator
from .model_selection import ModelSelector
from .preprocessing import DataPreprocessor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PackageName:
    def __init__(self, preprocess: bool = True, **preprocessor_kwargs):
        self.preprocess = preprocess
        # self.preprocess_args =
        self.preprocessor = (
            DataPreprocessor(**preprocessor_kwargs) if preprocess else None
        )
        self.model_selector = None
        # self.model_selector_kwargs = model_selector_kwargs

    def fit(self, X, y):
        if self.preprocess:
            X_train, y_train, X_test, y_test = self.preprocessor.fit_transform(X, y)
        else:
            X_train, y_train, X_test, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        self.model_selector = ModelSelector(X_train, y_train, X_test, y_test)
        best_model, best_params, best_score, fitted_models = (
            self.model_selector.compare_models()
        )

        log.info(f"Best model: {best_model.__class__.__name__}")
        log.info(f"Best parameters: {best_params}")
        log.info(f"Best score: {best_score:.4f}")

        evaluator = ModelEvaluator(fitted_models, X_test, y_test)
        results_df = evaluator.evaluate()
        evaluator.plot_results()

        return best_model, best_params, results_df
