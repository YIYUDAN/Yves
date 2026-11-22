import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import transforms

from src.models.wrn import WRN16_2
from src.fewshot.foreground import crop_foreground
from src.fewshot.metrics import save_confusion_matrix


def load_stats():
    with open("outputs/pretrain/train_mean_std.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def build_transform(mean, std):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def list_support(class_dir, k):
    files = sorted(class_dir.glob("*.jpg"))
    selected = []
    for index in range(1, k + 1):
        suffix = f"-{index:02d}.jpg"
        for path in files:
            if path.name.endswith(suffix):
                selected.append(path)
                break
    return selected


def extract_features(model, paths, transform, crop_mode, device):
    features = []
    with torch.no_grad():
        for path in paths:
            image = crop_foreground(Image.open(path).convert("RGB"), crop_mode)
            x = transform(image).unsqueeze(0).to(device)
            feat = model.forward_features(x)
            features.append(feat.squeeze(0).cpu())
    return torch.stack(features)


def l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--weights", default="outputs/pretrain/best_wrn16_2.pth")
    parser.add_argument("--crop-mode", default="grabcut")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    k = args.k
    data_root = Path("data/fewshot/G05/images")
    class_dirs = ["00", "01", "02", "03", "04"]
    out_dir = Path(args.out) if args.out else Path(f"outputs/fewshot/{k}shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_stats()
    transform = build_transform(mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WRN16_2(num_classes=30)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    support_paths = []
    support_labels = []
    for label, name in enumerate(class_dirs):
        paths = list_support(data_root / "support" / name, k)
        print(f"support {name} selected {len(paths)}")
        support_paths.extend(paths)
        support_labels.extend([label] * len(paths))

    query_paths = []
    query_labels = []
    query_names = []
    for label, name in enumerate(class_dirs):
        paths = sorted((data_root / "query" / name).glob("*.jpg"))
        query_paths.extend(paths)
        query_labels.extend([label] * len(paths))
        query_names.extend([p.name for p in paths])

    support_feat = l2_normalize(extract_features(model, support_paths, transform, args.crop_mode, device))
    query_feat = l2_normalize(extract_features(model, query_paths, transform, args.crop_mode, device))

    prototypes = []
    for label in range(len(class_dirs)):
        mask = torch.tensor([x == label for x in support_labels])
        prototypes.append(support_feat[mask].mean(dim=0))
    prototypes = l2_normalize(torch.stack(prototypes))

    scores = query_feat @ prototypes.t()
    preds = scores.argmax(dim=1).tolist()

    correct = sum(int(p == t) for p, t in zip(preds, query_labels))
    total = len(query_labels)
    acc = correct / total
    macro_f1 = f1_score(query_labels, preds, average="macro")
    cm = confusion_matrix(query_labels, preds, labels=list(range(len(class_dirs))))

    result = {
        "task": f"{k}shot",
        "weights": args.weights,
        "crop_mode": args.crop_mode,
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "class_dirs": class_dirs,
    }

    with open(out_dir / "baseline_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(cm.tolist(), class_dirs, out_dir / "baseline_confusion_matrix.png")

    with open(out_dir / "baseline_predictions.csv", "w", encoding="utf-8") as f:
        f.write("file_name,true_label,baseline_pred\n")
        for name, t, p in zip(query_names, query_labels, preds):
            f.write(f"{name},{t},{p}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()