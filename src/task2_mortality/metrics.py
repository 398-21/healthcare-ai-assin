"""
Metrics module for binary classification evaluation in healthcare ML.

Implements AUPRC, AUROC, sensitivity, specificity, PPV, bootstrap confidence intervals,
threshold finding, and the PhysioNet 2012 min(Se, PPV) metric.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)


def compute_metrics(y_true, y_score, threshold=None):
    """
    Compute AUPRC, AUROC, and optionally threshold-based metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores.
    threshold : float, optional
        Decision threshold for computing Se/PPV/Sp. If None, only AUPRC/AUROC returned.

    Returns
    -------
    dict
        Dictionary containing:
        - 'auprc': Area under precision-recall curve
        - 'auroc': Area under ROC curve
        - 'sensitivity': Sensitivity/recall at threshold (if threshold provided)
        - 'ppv': Positive predictive value at threshold (if threshold provided)
        - 'specificity': Specificity at threshold (if threshold provided)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Compute AUPRC using average_precision_score (not auc(recall, precision))
    auprc = average_precision_score(y_true, y_score)

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_score)

    metrics = {
        'auprc': auprc,
        'auroc': auroc
    }

    # If threshold provided, compute threshold-based metrics
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity (recall, true positive rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Positive predictive value (precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics.update({
            'sensitivity': sensitivity,
            'ppv': ppv,
            'specificity': specificity,
            'min_se_ppv': min(sensitivity, ppv),
        })

    return metrics


def bootstrap_ci(y_true, y_score, metric_fn, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence interval for a metric function.

    Uses stratified sampling to preserve class prevalence in each bootstrap sample.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores.
    metric_fn : callable
        Function that takes (y_true, y_score) and returns a scalar metric.
        Example: lambda y_t, y_s: roc_auc_score(y_t, y_s)
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    ci : float, default=0.95
        Confidence interval level (e.g., 0.95 for 95% CI).

    Returns
    -------
    tuple of (float, float, float)
        (point_estimate, lower_bound, upper_bound)
        - point_estimate: Metric computed on original data
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_samples = len(y_true)

    # Compute point estimate on original data
    point_estimate = metric_fn(y_true, y_score)

    # Separate indices by class for stratified sampling
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    # Store bootstrap metric values
    bootstrap_values = []

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        # Stratified bootstrap: sample with replacement from each class
        pos_sample = rng.choice(pos_indices, size=n_pos, replace=True)
        neg_sample = rng.choice(neg_indices, size=n_neg, replace=True)

        # Combine and shuffle
        boot_indices = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(boot_indices)

        y_true_boot = y_true[boot_indices]
        y_score_boot = y_score[boot_indices]

        try:
            metric_value = metric_fn(y_true_boot, y_score_boot)
            bootstrap_values.append(metric_value)
        except (ValueError, ZeroDivisionError):
            # Skip iterations where metric cannot be computed
            continue

    bootstrap_values = np.array(bootstrap_values)

    # Compute CI bounds using percentile method
    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_values, lower_percentile)
    upper_bound = np.percentile(bootstrap_values, upper_percentile)

    return point_estimate, lower_bound, upper_bound


def find_threshold_for_recall(y_true, y_score, target_recall=0.80):
    """
    Find the decision threshold that achieves approximately the target recall/sensitivity.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores.
    target_recall : float, default=0.80
        Desired recall/sensitivity level (between 0 and 1).

    Returns
    -------
    float
        Threshold value that achieves recall closest to target_recall.
        Returns the threshold where recall >= target_recall with minimal excess.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Get precision, recall, and thresholds from sklearn
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns (n_thresholds + 1) values for precision/recall
    # and n_thresholds values for thresholds. The last precision/recall values
    # correspond to threshold = 0 (all predicted as positive).
    # We need to align them properly.

    # Find thresholds where recall >= target_recall
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_indices) == 0:
        # No threshold achieves target recall, return minimum threshold
        return float(np.min(y_score))

    # Among valid thresholds, find the one with recall closest to target
    # (prefer slightly above target rather than well above)
    recall_diffs = np.abs(recalls[valid_indices] - target_recall)
    best_idx = valid_indices[np.argmin(recall_diffs)]

    return float(thresholds[best_idx])


def min_se_ppv(y_true, y_pred):
    """
    Compute the PhysioNet 2012 Challenge metric: min(sensitivity, PPV).

    This metric requires both high sensitivity and high precision, taking
    the minimum of the two as the final score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1), not probabilities.

    Returns
    -------
    float
        Minimum of sensitivity and positive predictive value.
        Returns 0.0 if either metric is undefined.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Positive predictive value (precision)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return min(sensitivity, ppv)
