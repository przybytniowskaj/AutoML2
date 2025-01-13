import os

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


class ModelEvaluator:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

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
        return self.results_df

    def plot_results(self):
        if not hasattr(self, "results_df"):
            raise ValueError("You need to run evaluate() before plotting results.")
        self.plot_roc_auc_curve()
        self.plot_all_confusion_matrices()

        return

    def plot_roc_auc_curve(self):
        plt.figure(figsize=(10, 6))

        for model in self.models:
            y_pred = model.predict(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred)
            plt.plot(fpr, tpr, label=f"{model.__class__.__name__} ROC ({auc:.2f})")

        plt.legend()
        plt.show()
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
        return

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

    def evaluate_to_html(self, training_summary):
        # Check if the training_summary is a DataFrame and not empty!:
        if training_summary is None or not isinstance(training_summary, pd.DataFrame) or training_summary.empty:
            raise ValueError("Can't produce a HTML report because training_summary should be a DataFrame and not empty.")

        # Save to training_summary table to HTML file:
        report_dir = os.getcwd() + "/mamut_report"
        os.makedirs(report_dir, exist_ok=True)
        training_summary.to_html(report_dir + "/training_summary.html")

        return training_summary.to_html()
