import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import transforms

from src.fewshot.foreground import crop_foreground
from src.fewshot.metrics import compute_metrics, save_confusion_matrix, save_predictions_csv
from src.models.wrn import WRN16_2


CLASS_DIRS = ["00", "01", "02", "03", "04"]
CLASS_NAMES = ["非洲野犬", "黑足鼬", "板条箱", "沙漏", "校车"]
K = 5
FEATURE_DIM = 128


def load_mean_std():
    with open("outputs/pretrain/train_mean_std.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def build_eval_transform(mean, std):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_train_transform(mean, std):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25),
        ]
    )


def load_images(paths, crop_mode):
    return [crop_foreground(Image.open(path).convert("RGB"), crop_mode) for path in paths]


def collect_samples(data_root):
    support_paths = []
    support_labels = []
    query_paths = []
    query_labels = []
    query_names = []

    for label, name in enumerate(CLASS_DIRS):
        support_dir = data_root / "support" / name
        files = sorted(support_dir.glob("*.jpg"))
        for index in range(1, K + 1):
            suffix = f"-{index:02d}.jpg"
            for path in files:
                if path.name.endswith(suffix):
                    support_paths.append(path)
                    support_labels.append(label)
                    break

        query_files = sorted((data_root / "query" / name).glob("*.jpg"))
        query_paths.extend(query_files)
        query_labels.extend([label] * len(query_files))
        query_names.extend([p.name for p in query_files])

    return support_paths, support_labels, query_paths, query_labels, query_names


class CosineHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.01)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, features):
        feat = features / features.norm(dim=1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=1, keepdim=True)
        return self.log_scale.exp() * (feat @ weight.t())


def load_base_state(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    return state["model"]


def prototype_weight(model, images, labels, transform, device):
    model.eval()
    features = extract_features(model, images, transform, device)
    features = features / features.norm(dim=1, keepdim=True)
    prototypes = []
    for label in range(len(CLASS_DIRS)):
        mask = torch.tensor([item == label for item in labels])
        prototypes.append(features[mask].mean(dim=0))
    weight = torch.stack(prototypes)
    return weight / weight.norm(dim=1, keepdim=True)


def finetune(base_state, support_images, support_labels, seed, epochs, lr, device, transform, head_init, log_rows):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = WRN16_2(num_classes=30)
    model.load_state_dict(base_state)
    model.to(device)

    head = CosineHead(FEATURE_DIM, len(CLASS_DIRS)).to(device)
    if head_init == "prototype":
        head.weight.data.copy_(prototype_weight(model, support_images, support_labels, transform, device))
    model.train()

    labels = torch.tensor(support_labels)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": lr},
            {"params": head.parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, epochs + 1):
        batch = torch.stack([transform(image) for image in support_images]).to(device)
        target = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = head(model.forward_features(batch))
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        train_acc = (logits.argmax(dim=1) == target).float().mean().item()
        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append([seed, epoch, loss.item(), train_acc, current_lr])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"seed {seed} epoch {epoch} loss {loss.item():.4f} train_acc {train_acc:.4f}")

    model.eval()
    return model


def encode(model, batch, mode):
    x = model.relu(model.bn1(model.conv1(batch)))
    x = model.layer1(x)
    stage2 = model.layer2(x)
    stage3 = model.layer3(stage2)
    gap3 = stage3.mean(dim=(2, 3))
    if mode == "global":
        return gap3
    if mode == "global_max":
        return torch.cat([gap3, stage3.amax(dim=(2, 3))], dim=1)
    if mode == "multi_layer":
        return torch.cat([gap3, stage2.mean(dim=(2, 3))], dim=1)
    if mode == "multi_layer_max":
        return torch.cat(
            [gap3, stage3.amax(dim=(2, 3)), stage2.mean(dim=(2, 3)), stage2.amax(dim=(2, 3))],
            dim=1,
        )
    raise ValueError(mode)


def extract_features(model, images, transform, device, mode="global", tta=False, batch_size=50):
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            batch = torch.stack([transform(image) for image in chunk]).to(device)
            feat = encode(model, batch, mode)
            if tta:
                feat = (feat + encode(model, torch.flip(batch, dims=[3]), mode)) / 2
            features.append(feat.cpu())
    return torch.cat(features)


def fit_logistic_regression(support_features, support_labels, query_features, c_value):
    scaler = StandardScaler().fit(support_features.numpy())
    classifier = LogisticRegression(C=c_value, max_iter=3000)
    classifier.fit(scaler.transform(support_features.numpy()), np.array(support_labels))
    return classifier.predict_proba(scaler.transform(query_features.numpy()))


def support_loo(base_state, support_images, support_labels, crop_mode, seeds, epochs, lr, c_values, head_init, device, transform):
    records = []
    for seed in seeds:
        for fold in range(K):
            train_index = [index for index in range(len(support_labels)) if index % K != fold]
            hold_index = [index for index in range(len(support_labels)) if index % K == fold]
            train_images = [support_images[index] for index in train_index]
            train_labels = [support_labels[index] for index in train_index]
            hold_images = [support_images[index] for index in hold_index]
            hold_labels = [support_labels[index] for index in hold_index]

            print(f"cross validation crop {crop_mode} seed {seed} fold {fold}")
            model = finetune(base_state, train_images, train_labels, seed, epochs, lr, device, transform, head_init, [])
            train_features = extract_features(model, train_images, transform, device)
            hold_features = extract_features(model, hold_images, transform, device)

            for c_value in c_values:
                probs = fit_logistic_regression(train_features, train_labels, hold_features, c_value)
                preds = probs.argmax(axis=1).tolist()
                correct = sum(int(p == t) for p, t in zip(preds, hold_labels))
                hit = [float(probs[index, label]) for index, label in enumerate(hold_labels)]
                records.append(
                    {
                        "crop_mode": crop_mode,
                        "seed": seed,
                        "fold": fold,
                        "c_value": c_value,
                        "correct": correct,
                        "total": len(hold_labels),
                        "true_prob_sum": sum(hit),
                    }
                )
    return records


def summarize_loo(records):
    totals = {}
    for record in records:
        key = (record["crop_mode"], record["c_value"])
        item = totals.setdefault(key, {"correct": 0, "total": 0, "true_prob_sum": 0.0})
        item["correct"] += record["correct"]
        item["total"] += record["total"]
        item["true_prob_sum"] += record["true_prob_sum"]

    rows = []
    for (crop_mode, c_value), item in totals.items():
        rows.append(
            {
                "crop_mode": crop_mode,
                "c_value": c_value,
                "correct": item["correct"],
                "total": item["total"],
                "accuracy": item["correct"] / item["total"],
                "mean_true_prob": item["true_prob_sum"] / item["total"],
            }
        )
    return rows


def save_misclassified_csv(path, query_names, true_labels, preds, probs):
    rows = []
    for index, (name, true_label, pred_label) in enumerate(zip(query_names, true_labels, preds)):
        if true_label == pred_label:
            continue
        rows.append(
            {
                "file_name": name,
                "true_label": true_label,
                "true_name": CLASS_NAMES[true_label],
                "pred_label": pred_label,
                "pred_name": CLASS_NAMES[pred_label],
                "true_prob": float(probs[index, true_label]),
                "pred_prob": float(probs[index, pred_label]),
            }
        )

    rows.sort(key=lambda item: item["true_prob"])
    fieldnames = [
        "file_name",
        "true_label",
        "true_name",
        "pred_label",
        "pred_name",
        "true_prob",
        "pred_prob",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_misclassified_grid(path, rows, query_paths, query_names):
    path_by_name = {name: path for path, name in zip(query_paths, query_names)}
    selected = rows[:6]
    if not selected:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, row in zip(axes.ravel(), selected):
        image = Image.open(path_by_name[row["file_name"]]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{row['file_name']}\ntrue {row['true_label']} pred {row['pred_label']}\n"
            f"p(true)={row['true_prob']:.2f}",
            fontsize=9,
        )
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_train_curve(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = sorted({row[1] for row in rows})
    loss = []
    acc = []
    for epoch in epochs:
        items = [row for row in rows if row[1] == epoch]
        loss.append(sum(row[2] for row in items) / len(items))
        acc.append(sum(row[3] for row in items) / len(items))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, loss)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("adaptation loss")
    axes[1].plot(epochs, acc)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("support-set train accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/pretrain/best_wrn16_2.pth")
    parser.add_argument("--seeds", default="2026,1234,777")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--crop-mode", default="grabcut")
    parser.add_argument("--crop-modes", default="none,center,grabcut")
    parser.add_argument("--head-init", default="random")
    parser.add_argument("--feature-modes", default="global")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--c-values", default="0.1,1.0,10.0")
    parser.add_argument("--loo", action="store_true")
    parser.add_argument("--out", default="outputs/fewshot/5shot")
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",")]
    data_root = Path("data/fewshot/G05/images")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_mean_std()
    eval_transform = build_eval_transform(mean, std)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_state = load_base_state(args.weights)

    support_paths, support_labels, query_paths, query_labels, query_names = collect_samples(data_root)
    crop_mode = args.crop_mode
    c_value = args.c_value
    selection_rows = []

    if args.loo:
        c_values = [float(item) for item in args.c_values.split(",")]
        records = []
        for mode in args.crop_modes.split(","):
            print(f"support cross validation crop mode {mode}")
            mode_images = load_images(support_paths, mode)
            records.extend(
                support_loo(
                    base_state,
                    mode_images,
                    support_labels,
                    mode,
                    seeds,
                    args.epochs,
                    args.lr,
                    c_values,
                    args.head_init,
                    device,
                    eval_transform,
                )
            )

        selection_rows = summarize_loo(records)
        best = max(selection_rows, key=lambda row: (row["accuracy"], row["mean_true_prob"]))
        crop_mode = best["crop_mode"]
        c_value = best["c_value"]

        with open(out_dir / "support_cv.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["crop_mode", "c_value", "correct", "total", "accuracy", "mean_true_prob"])
            for row in selection_rows:
                writer.writerow(
                    [
                        row["crop_mode"],
                        row["c_value"],
                        row["correct"],
                        row["total"],
                        f"{row['accuracy']:.4f}",
                        f"{row['mean_true_prob']:.4f}",
                    ]
                )

        with open(out_dir / "support_cv_records.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["crop_mode", "seed", "fold", "c_value", "correct", "total", "true_prob_sum"])
            for record in records:
                writer.writerow(
                    [
                        record["crop_mode"],
                        record["seed"],
                        record["fold"],
                        record["c_value"],
                        record["correct"],
                        record["total"],
                        f"{record['true_prob_sum']:.4f}",
                    ]
                )

        print(f"selected crop_mode {crop_mode} c_value {c_value}")

    support_images = load_images(support_paths, crop_mode)
    query_images = load_images(query_paths, crop_mode)
    print(f"support {len(support_images)} query {len(query_images)} device {device} crop {crop_mode}")

    log_rows = []
    feature_modes = args.feature_modes.split(",")
    mode_probs = {mode: [] for mode in feature_modes}
    mode_seeds = {mode: [] for mode in feature_modes}

    for seed in seeds:
        model = finetune(
            base_state,
            support_images,
            support_labels,
            seed,
            args.epochs,
            args.lr,
            device,
            eval_transform,
            args.head_init,
            log_rows,
        )

        for mode in feature_modes:
            support_features = extract_features(model, support_images, eval_transform, device, mode, args.tta)
            query_features = extract_features(model, query_images, eval_transform, device, mode, args.tta)
            probs = fit_logistic_regression(
                support_features,
                support_labels,
                query_features,
                c_value,
            )
            mode_probs[mode].append(probs)

            seed_preds = probs.argmax(axis=1).tolist()
            seed_correct = sum(int(p == t) for p, t in zip(seed_preds, query_labels))
            mode_seeds[mode].append(
                {"seed": seed, "correct": seed_correct, "accuracy": seed_correct / len(query_labels)}
            )
            print(f"seed {seed} feature {mode} accuracy {seed_correct / len(query_labels):.4f}")

    with open(out_dir / "adapt_train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "epoch", "loss", "train_acc", "lr"])
        writer.writerows(log_rows)
    save_train_curve(log_rows, out_dir / "adapt_train_curve.png")

    summary_rows = []
    for mode in feature_modes:
        probs = np.mean(mode_probs[mode], axis=0)
        preds = probs.argmax(axis=1).tolist()
        metrics = compute_metrics(query_labels, preds, CLASS_DIRS)
        mode_dir = out_dir if len(feature_modes) == 1 else out_dir / f"feature_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "task": "5shot",
            "method": f"foreground crop, {args.head_init}-initialized cosine head, support-set finetune, logistic regression readout",
            "weights": args.weights,
            "seeds": seeds,
            "epochs": args.epochs,
            "lr": args.lr,
            "crop_mode": crop_mode,
            "c_value": c_value,
            "feature_mode": mode,
            "tta": args.tta,
            "selection": selection_rows,
            "correct": metrics["correct"],
            "total": metrics["total"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "class_dirs": CLASS_DIRS,
            "class_names": CLASS_NAMES,
            "seed_results": mode_seeds[mode],
        }

        with open(mode_dir / "adapt_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        save_predictions_csv(
            mode_dir / "adapt_predictions.csv", query_names, query_labels, preds, probs, CLASS_DIRS
        )
        save_confusion_matrix(
            metrics["confusion_matrix"], CLASS_DIRS, mode_dir / "adapt_confusion_matrix.png"
        )
        rows = save_misclassified_csv(
            mode_dir / "misclassified_samples.csv", query_names, query_labels, preds, probs
        )
        save_misclassified_grid(mode_dir / "misclassified_grid.png", rows, query_paths, query_names)

        summary_rows.append(
            {
                "feature_mode": mode,
                "tta": args.tta,
                "correct": metrics["correct"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
        print(f"feature {mode} accuracy {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f}")

    if len(feature_modes) > 1:
        with open(out_dir / "feature_ablation.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feature_mode", "tta", "correct", "accuracy", "macro_f1"])
            for row in summary_rows:
                writer.writerow(
                    [
                        row["feature_mode"],
                        row["tta"],
                        row["correct"],
                        f"{row['accuracy']:.4f}",
                        f"{row['macro_f1']:.4f}",
                    ]
                )


if __name__ == "__main__":
    main()
