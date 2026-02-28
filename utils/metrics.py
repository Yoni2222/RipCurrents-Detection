import numpy as np

def calculate_metrics(tp, fp, fn, total_samples, correct_preds):
    accuracy = correct_preds / (total_samples + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return accuracy, precision, recall, f1

def calculate_full_metrics(probs, labels, threshold):
    preds = (probs > threshold).astype(int)
    labels = labels.astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    accuracy = (preds == labels).mean()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, precision, recall, f1

def find_optimal_threshold_by_acc(probs, labels):
    best_acc = 0.0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.95, 0.01):
        acc, _, _, _ = calculate_full_metrics(probs, labels, t)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc