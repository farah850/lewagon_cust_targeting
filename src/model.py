from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,fbeta_score,roc_auc_score,roc_curve,precision_score,recall_score
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
import numpy as np

def build_logreg_pipeline(preprocessor, max_iter=1000, random_state=42):
    """
    Creates a pipeline with preprocessing and logistic regression classifier.

    Args:
        preprocessor (Transformer): A fitted or unfitted preprocessing transformer.
        max_iter (int): Maximum number of iterations for LogisticRegression.
        random_state (int): Random state for reproducibility.

    Returns:
        Pipeline: A scikit-learn pipeline.
    """
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iter, random_state=random_state))
    ])
    return pipeline


def evaluate_model(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_proba: Optional[Union[np.ndarray, list]] = None,
    beta: float = 1.0,
    average: str = 'binary',
    plot_roc: bool = False
) -> Dict[str, Union[float, str]]:
    """
    Evaluates a classification model using common metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like, optional): Predicted probabilities, required for ROC AUC and ROC curve.
        beta (float, optional): Weight of recall in the F-beta score. Defaults to 1.0 (F1 score).
        average (str, optional): Scoring method for multi-class problems. One of 'binary', 'macro', 'micro', or 'weighted'.
        plot_roc (bool, optional): Whether to display the ROC curve.

    Returns:
        dict: Dictionary containing various evaluation metrics.
    """
    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average=average),
        f'F{beta}-Score': fbeta_score(y_true, y_pred, beta=beta, average=average)
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            results['ROC AUC'] = roc_auc
            if plot_roc:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            results['ROC AUC'] = f'Error: {str(e)}'

    return results
