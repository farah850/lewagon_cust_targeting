from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,fbeta_score,roc_auc_score,roc_curve,precision_score,recall_score, precision_recall_curve, PrecisionRecallDisplay,average_precision_score
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
import numpy as np

def build_rf_pipeline(preprocessor, n_estimators=100, max_depth=None, min_samples_split=2,
                      min_samples_leaf=1, max_features='sqrt', random_state=42):
    """
    Creates a pipeline with preprocessing and random forest classifier.

    Args:
        preprocessor (Transformer): A fitted or unfitted preprocessing transformer.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree. If None, nodes are expanded until
                        all leaves are pure or contain less than min_samples_split samples.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (str or int): Number of features to consider when looking for the best split.
        random_state (int): Random state for reproducibility.

    Returns:
        Pipeline: A scikit-learn pipeline.
    """
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        ))
    ])
    return pipeline

def evaluate_rf_model(
    y_true: Union[np.ndarray, list],
    y_pred: Optional[Union[np.ndarray, list]] = None,
    y_proba: Optional[Union[np.ndarray, list]] = None,
    threshold: float = 0.10,
    beta: float = 1.0,
    average: str = 'binary',
    plot_roc: bool = False,
    plot_pr: bool = False,
    min_recall: float = None
) -> Dict[str, Union[float, str, dict]]:
    """
    Evaluates a Random Forest classification model using a custom threshold if probabilities are provided.
    Can also find the best threshold given a minimum recall requirement.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels (optional if y_proba is provided).
        y_proba: Predicted probabilities for the positive class (optional).
        threshold: Threshold for converting probabilities to predictions.
        beta: Beta parameter for F-beta score.
        average: Type of averaging performed on the data ('binary', 'micro', 'macro', 'weighted').
        plot_roc: Whether to plot ROC curve.
        plot_pr: Whether to plot Precision-Recall curve.
        min_recall: Minimum recall requirement for threshold optimization.

    Returns:
        Dict: Dictionary containing various evaluation metrics.
    """
    if y_proba is not None:
        y_pred = (np.array(y_proba) >= threshold).astype(int)

    results = {
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average=average),
        f'F{beta}-Score': fbeta_score(y_true, y_pred, beta=beta, average=average)
    }

    if y_proba is not None:
        try:
            # ROC AUC
            roc_auc = roc_auc_score(y_true, y_proba)
            results['ROC AUC'] = roc_auc

            if plot_roc:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Random Forest - ROC Curve")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            # Precision-Recall
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            results['Average Precision'] = avg_precision

            # Find best threshold for minimum recall
            best_data = None
            if min_recall is not None:
                best_acc = -1
                for p, r, t in zip(precision[:-1], recall[:-1], pr_thresholds):
                    if r >= min_recall:
                        preds = (np.array(y_proba) >= t).astype(int)
                        acc = accuracy_score(y_true, preds)
                        if acc > best_acc:
                            best_acc = acc
                            best_data = {
                                "Best Threshold": t,
                                "Precision": p,
                                "Recall": r,
                                "Accuracy": acc
                            }
                results['Best Threshold for Min Recall'] = best_data

            if plot_pr:
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.2f})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Random Forest - Precision-Recall Curve")
                plt.grid(True)

                # Mark the best threshold point
                if min_recall is not None and best_data is not None:
                    plt.scatter(best_data["Recall"], best_data["Precision"],
                                color="red", s=80, zorder=5, label="Best ≥ Min Recall")
                    plt.annotate(f"t={best_data['Best Threshold']:.2f}\nAcc={best_data['Accuracy']:.2f}",
                                 (best_data["Recall"], best_data["Precision"]),
                                 textcoords="offset points", xytext=(5, -10), fontsize=8)

                plt.legend()
                plt.tight_layout()
                plt.show()

        except Exception as e:
            results['ROC AUC'] = f'Error: {str(e)}'

    return results

def get_feature_importance(pipeline, feature_names=None, top_n=10, plot=True):
    """
    Extract and optionally plot feature importance from a trained Random Forest pipeline.

    Args:
        pipeline: Trained sklearn pipeline with RandomForestClassifier.
        feature_names: List of feature names (optional).
        top_n: Number of top features to display.
        plot: Whether to create a bar plot of feature importances.

    Returns:
        Dict: Dictionary with feature names/indices as keys and importance scores as values.
    """
    if not hasattr(pipeline, 'named_steps') or 'classifier' not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a 'classifier' step")

    classifier = pipeline.named_steps['classifier']
    if not hasattr(classifier, 'feature_importances_'):
        raise ValueError("Classifier must have feature_importances_ attribute")

    importances = classifier.feature_importances_

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]

    # Create importance dictionary and sort by importance
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    if plot:
        top_features = sorted_importance[:top_n]
        features, scores = zip(*top_features)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, (feature, score) in enumerate(top_features):
            plt.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    return dict(sorted_importance)
