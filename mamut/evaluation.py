import os
import time
from datetime import datetime
from typing import Callable, List
import base64
import platform

import numpy as np
import optuna
import psutil

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from mamut.preprocessing.handlers import handle_outliers
from mamut.utils.utils import preprocessing_steps, model_param_dict


def _get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _generate_experiment_setup_table():
    system_info = {
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "System": platform.system(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Python Version": platform.python_version(),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "CPU Cores": psutil.cpu_count(logical=True),
    }

    # Convert the system information to a DataFrame
    df = pd.DataFrame(system_info.items(), columns=["Attribute", "Value"])

    # Convert the DataFrame to an HTML table
    html_table = df.to_html(index=False)

    return html_table

def _generate_dataset_overview(X: pd.DataFrame, y: pd.Series) -> (List[int], pd.DataFrame, pd.DataFrame):
    # Dataset shape
    n_observations, n_features = X.shape

    # Calculate number of rows with any missing values:
    n_rows_missing = X.isnull().any(axis=1).sum()

    # Calculate the number of outliers according to IsolationForest method:
    # TODO: Check if correct
    _, y_new, _ = handle_outliers(X, y, X.columns)
    n_outliers = len(y) - len(y_new)

    dataset_basic_list = [n_observations, n_features, n_rows_missing, n_outliers]

    # Feature summary
    feature_summary = X.dtypes.reset_index()
    feature_summary.columns = ["Feature", "Data Type"]
    feature_summary["Type"] = feature_summary["Data Type"].apply(
        lambda dt: "Categorical" if dt == "object" or isinstance(dt, pd.CategoricalDtype) else "Numerical")
    if len(feature_summary) > 10:
        feature_summary = feature_summary.head(10)
    feature_summary = feature_summary[["Feature", "Type", "Data Type"]]

    # Class distribution (if target column is provided)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    class_distribution = y.value_counts().reset_index()
    class_distribution.columns = ["Class", "Count"]

    return dataset_basic_list, feature_summary, class_distribution

def _generate_preprocessing_steps_list(steps: List[str]) -> str:
    # Initialize a dictionary to hold categorized steps
    categorized_steps = {}

    # Categorize each step
    for step in steps:
        if step in preprocessing_steps:
            category, description = preprocessing_steps[step]
            if category not in categorized_steps:
                categorized_steps[category] = []
            categorized_steps[category].append(f"<strong>{step}</strong>: {description}")

    # Generate HTML unordered list with some styling
    html_prep_list = ""
    for category, tools in categorized_steps.items():
        html_prep_list += f"<li style='padding-left: 10px;'><strong>{category}</strong><ul style='list-style-type: '&#x2192'; margin-left: 20px;'>"
        for tool in tools:
            html_prep_list += f"<li>{tool}</li>"
        html_prep_list += "</ul></li>"

    return html_prep_list

def _generate_models_list(excluded_models: List[str]) -> List[str]:
    # Get all available models
    all_models = model_param_dict.keys()
    # Remove excluded models
    available_models = [model for model in all_models if model not in excluded_models]

    return available_models


class ModelEvaluator:

    report_template_path : str = os.path.join(os.path.dirname(__file__), "utils")

    def __init__(self,
                 models: dict,
                 # X_test and y_test are preprocessed. X and y are not.
                 X_test : np.ndarray,
                 y_test : np.ndarray,
                 X : pd.DataFrame,
                 y : pd.Series,
                 optimizer: str,
                 n_trials: int,
                 metric: str,
                 studies: dict,
                 excluded_models : List[str] = None,
                 ):
        self.models = models
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.optimizer = optimizer
        self.n_trials = n_trials
        self.metric = metric
        self.studies = studies
        self.excluded_models = excluded_models if excluded_models else []

        self.report_output_path = os.path.join(os.getcwd(), "mamut_report")
        self.plot_output_path = os.path.join(self.report_output_path, "plots")

        # Create the report directory it doesn't exist:
        os.makedirs(self.report_output_path, exist_ok=True)
        os.makedirs(self.plot_output_path, exist_ok=True)

        sns.set_context("notebook", font_scale=1.1)
        plt.style.use("fivethirtyeight")
        # Set background color of all plots to #f0f8ff;
        plt.rcParams["axes.facecolor"] = "#f0f8ff"
        plt.rcParams["figure.facecolor"] = "#f0f8ff"
        # Set border color to skyblue
        plt.rcParams["axes.edgecolor"] = "#007bb5"
        plt.rcParams["figure.edgecolor"] = "#007bb5"

    def evaluate(self, training_summary : pd.DataFrame):
        return self.evaluate_to_html(training_summary)

    def plot_results(self):
        if not hasattr(self, "results_df"):
            raise ValueError("You need to run evaluate() before plotting results.")
        self._plot_roc_auc_curve()
        self.plot_all_confusion_matrices()

        return

    def _plot_roc_auc_curve(self, training_summary: pd.DataFrame, show: bool = False, save: bool = True) -> None:
        plt.figure(figsize=(10, 6))
        top_3_models = training_summary["Model"].head(3).to_numpy()

        for model_name in top_3_models:
            model = next(m for m in self.models.values() if m.__class__.__name__ == model_name)
            y_pred = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred)
            plt.plot(fpr, tpr, label=f"{model_name} ROC ({auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()

        if show:
            plt.show()
        if save:
            plt.savefig(os.path.join(self.plot_output_path, "roc_auc_curve.png"), format="png", bbox_inches="tight")

        return

        # y_test_bin = label_binarize(self.y_test, classes=list(set(self.y_test)))
        # n_classes = y_test_bin.shape[1]

        # plt.figure(figsize=(10, 8))

        # for model in self.models:
        #     y_score = model.predict(self.X_test)
        #     fpr = dict()
        #     tpr = dict()
        #     roc_auc = dict()
        #     for i in range(n_classes):
        #         fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        #         roc_auc[i] = auc(fpr[i], tpr[i])

        #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        #     for i, color in zip(range(n_classes), colors):
        #         plt.plot(fpr[i], tpr[i], color=color, lw=2,
        #             label=f'ROC curve of class {i} for {model.__class__.__name__} (area = {roc_auc[i]:0.2f})')

        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()

    def _plot_confusion_matrices(self, training_summary: pd.DataFrame, show: bool = False, save: bool = True) -> None:
        fig = plt.figure(figsize=(18, 5))
        top_3_models = training_summary["Model"].head(3).to_numpy()
        # plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(1, 3, wspace=0.4)  # Create a grid with 1 row and 3 columns, with space between plots

        for i, model_name in enumerate(top_3_models):
            model = next(m for m in self.models.values() if m.__class__.__name__ == model_name)
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            ax = fig.add_subplot(gs[i])
            #plt.subplot(1, 3, i + 1)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            plt.title(f"{model_name}", fontsize=14)
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)

        #plt.subplots_adjust(wspace=0.4)  # Add space between plots
        plt.tight_layout()

        if show:
            plt.show()
        if save:
            plt.savefig(os.path.join(self.plot_output_path, "confusion_matrices.png"), format="png",
                        bbox_inches="tight")

        return

    def _plot_hyperparameter_tuning_history(self, training_summary: pd.DataFrame, show: bool = False,
                                            save: bool = True) -> None:
        top_3_models = training_summary["Model"].head(3).to_numpy()

        for i, model_name in enumerate(top_3_models):
            study = self.studies.get(model_name)
            if study:
                plt.figure(figsize=(6, 5), facecolor="#f0f8ff")
                ax = optuna.visualization.matplotlib.plot_optimization_history(study)
                ax.set_facecolor("#f0f8ff")
                ax.spines['top'].set_color("#007bb5")
                ax.spines['right'].set_color("#007bb5")
                ax.spines['bottom'].set_color("#007bb5")
                ax.spines['left'].set_color("#007bb5")
                ax.grid(color='grey')  # Change grid color to grey
                ax.legend().set_visible(False)  # Remove legend
                plt.title(f"{model_name} Tuning History", fontsize=14)
                plt.xlabel("Trial", fontsize=12)
                plt.ylabel(f"{self.metric} Value", fontsize=12)
                plt.tight_layout()

                if show:
                    plt.show()
                if save:
                    plt.savefig(os.path.join(self.plot_output_path, f"hyperparameter_tuning_history_{i + 1}.png"),
                                format="png", bbox_inches="tight")
                plt.close()

        return

    def evaluate_to_html(self, training_summary : pd.DataFrame,
                         ):
        # Check if the training_summary is a DataFrame and not empty!:
        if (
            training_summary is None
            or not isinstance(training_summary, pd.DataFrame)  # noqa
            or training_summary.empty  # noqa
        ):
            raise ValueError(
                "Can't produce a HTML report because training_summary should be a DataFrame and not empty."
            )

        # Preprocess the training_summary DataFrame:
        training_summary = training_summary.rename(columns={
            "model": "Model",
            "accuracy_score": "Accuracy",
            "balanced_accuracy_score": "Balanced Accuracy",
            "precision_score": "Precision",
            "recall_score": "Recall",
            "f1_score": "F1 Score",
            "jaccard_score": "Jaccard Score",
            "roc_auc_score": "ROC AUC",
            "duration": "Training Time [s]",
        })
        # Sort the training_summary DataFrame by the score_metric column
        training_summary = (training_summary.sort_values(by=training_summary.columns[1], ascending=False).reset_index(drop=True))

        # Apply the style to the DataFrame
        styled_training_summary = training_summary.style.apply(_highlight_first_cell, axis=1)

        # Transform summary to HTML:
        training_summary_html = styled_training_summary.to_html()

        # Transform the header image to base64:
        image_header_path = os.path.join(self.report_template_path, "mamut_header.png")
        base64_image = _get_base64_image(image_header_path)

        # Calculate Dataset Overview:
        dataset_basic_list, feature_summary, class_distribution = _generate_dataset_overview(self.X, self.y)

        # Create and save roc_auc_curve as .png file:
        self._plot_roc_auc_curve(training_summary)
        self._plot_confusion_matrices(training_summary)
        self._plot_hyperparameter_tuning_history(training_summary)

        # Load the Jinja2 template placed in report_template_path:
        env = Environment(loader=FileSystemLoader(self.report_template_path))
        template = env.get_template("report_template.html")

        # Render the template with the training_summary and save the HTML file
        time_signature = str(time.strftime(" %d %B %Y, %I:%M %p", time.localtime()))

        html_content = template.render(
            time_signature=time_signature,
            training_summary=training_summary_html,
            image_header=base64_image,
            experiment_setup=_generate_experiment_setup_table(),
            models_evaluated=_generate_models_list(self.excluded_models),
            optimizer="Tree-structured Parzen Estimator" if self.optimizer == "bayes" else "Random Search",
            metric=self.metric,
            n_trials=self.n_trials,
            best_model=training_summary.iloc[0]["Model"],
            basic_dataset_info=dataset_basic_list,
            feature_summary=feature_summary.to_html(index=False),
            class_distribution=class_distribution.to_html(index=False),
            # TODO: Get preprocessing steps from Preprocessor
            preprocessing_list=_generate_preprocessing_steps_list(["SimpleImputer", "StandardScaler"]),
        )

        time_signature = datetime.strptime(time_signature.strip(), "%d %B %Y, %I:%M %p").strftime("%d-%m-%Y_%H-%M")
        with open(os.path.join(self.report_output_path, f"report_{time_signature}.html"), "w") as f:
            f.write(html_content)

        return html_content

def _highlight_first_cell(s):
    return ['background-color: yellow' if (i == 0 and s.name == 0) or (i == 1 and s.name == 0) else ''
            for i in range(len(s))]