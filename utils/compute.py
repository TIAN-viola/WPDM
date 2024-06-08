from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def seq_accuracy(preds, labels):
    acc = []
    for idx, pred in enumerate(preds):
        acc.append((pred == labels[idx]).mean())
    return acc.mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def all_metrics(preds, labels, average=None):
    acc = simple_accuracy(preds, labels)
    if average:
        f1 = f1_score(y_true=labels, y_pred=preds, average=average)
        pre = precision_score(y_true=labels, y_pred=preds, average=average)
        rec = recall_score(y_true=labels, y_pred=preds, average=average)
    else:
        f1 = f1_score(y_true=labels, y_pred=preds)
        pre = precision_score(y_true=labels, y_pred=preds)
        rec = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }


def compute_metrics(preds, labels, average=None):
    assert len(preds) == len(labels)
    return all_metrics(preds, labels, average)