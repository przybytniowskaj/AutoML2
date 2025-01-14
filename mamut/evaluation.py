import os
import time
from typing import Callable
import base64

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

def _get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class ModelEvaluator:

    report_template_path : str = os.path.join(os.path.dirname(__file__), "utils")

    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

        self.report_output_path = os.path.join(os.getcwd(), "mamut_report")
        self.plot_output_path = os.path.join(self.report_output_path, "plots")

        # Create the report directory it doesn't exist:
        os.makedirs(self.report_output_path, exist_ok=True)
        os.makedirs(self.plot_output_path, exist_ok=True)

    def evaluate(self):
        results = []
        for model in self.models:
            y_pred = model.predict(self.X_test)
            results.append(
                {
                    "Model": model.__class__.__name__,
                    "Accuracy": accuracy_score(self.y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(self.y_test, y_pred),
                    "Precision": precision_score(
                        self.y_test, y_pred, average="weighted"
                    ),
                    "Recall": recall_score(self.y_test, y_pred, average="weighted"),
                    "F1 Score": f1_score(self.y_test, y_pred, average="weighted"),
                    "Jackard Score": jaccard_score(
                        self.y_test, y_pred, average="weighted"
                    ),
                }
            )

        self.results_df = pd.DataFrame(results)
        # TODO: Najprawodopodobniej evaluator nie musi zwracać DF, bo to jest w training_report
        return self.results_df

    def plot_results(self):
        if not hasattr(self, "results_df"):
            raise ValueError("You need to run evaluate() before plotting results.")
        self._plot_roc_auc_curve()
        self.plot_all_confusion_matrices()

        return

    def _plot_roc_auc_curve(self, show: bool = False, save: bool = True) -> None:
        plt.figure(figsize=(10, 6))

        for model in self.models:
            y_pred = model.predict(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred)
            plt.plot(fpr, tpr, label=f"{model.__class__.__name__} ROC ({auc:.2f})")

        plt.legend()
        plt.title("ROC AUC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
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


    def plot_all_confusion_matrices(self):
        if not hasattr(self, "results_df"):
            raise ValueError(
                "You need to run evaluate() before plotting the confusion matrices."
            )

        n_cols = 2
        n_rows = (len(self.models) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
        axes = axes.flatten()

        for ax, model in zip(axes, self.models):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix for {model.__class__.__name__}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        for ax in axes[len(self.models) :]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.show()
        return

    def evaluate_to_html(self, training_summary : pd.DataFrame,
                         score_metric : Callable):
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
        training_summary = training_summary.sort_values(by=training_summary.columns[1], ascending=False)

        # Transform summary to HTML:
        training_summary_html = training_summary.to_html()

        # Transform the header image to base64:
        image_header_path = os.path.join(self.report_template_path, "mamut_header.png")
        base64_image = _get_base64_image(image_header_path)

        # Create and save roc_auc_curve as .png file:
        self._plot_roc_auc_curve()


        # Load the Jinja2 template placed in report_template_path:
        env = Environment(loader=FileSystemLoader(self.report_template_path))
        template = env.get_template("report_template.html")

        # Render the template with the training_summary and save the HTML file
        time_signature = str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        html_content = template.render(time_signature=time_signature, training_summary=training_summary_html, image_header=base64_image)

        with open(os.path.join(self.report_output_path, "report.html"), "w") as f:
            f.write(html_content)

        return html_content
