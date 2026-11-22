import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(labels, preds, class_names):
    total = len(labels)
    correct = sum(int(p == t) for p, t in zip(preds, labels))
    acc = correct / total
    macro_f1 = f1_score(labels, preds, average="macro")
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(labels, preds, labels=class_ids)
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, class_names, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_predictions_csv(path, names, labels, preds, probs, class_names):
    with open(path, "w", encoding="utf-8") as f:
        f.write("file_name,true_label,true_name,pred_label,pred_name,true_prob\n")
        for name, t, p, prob in zip(names, labels, preds, probs):
            f.write(f"{name},{t},{class_names[t]},{p},{class_names[p]},{prob[t]:.6f}\n")
