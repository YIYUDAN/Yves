import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.pretrain_dataset import load_split_samples
from src.models.wrn import WRN16_2
from src.train_pretrain import evaluate, train_one_epoch


class CachedDataset(Dataset):
    def __init__(self, data_root, split, transform):
        samples = load_split_samples(data_root, split)
        self.images = []
        self.labels = []
        for path, label in samples:
            image = Image.open(path).convert("RGB")
            self.images.append(image.resize((256, 256)))
            self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def build_transforms(mean, std, is_train):
    if is_train:
        ops = [
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2),
        ]
    else:
        ops = [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(ops)


def main():
    with open("configs/pretrain_retrained.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = config["weight_tag"]

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = True

    with open(output_dir / "train_mean_std.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    train_dataset = CachedDataset(
        config["data_root"],
        "base_train",
        build_transforms(stats["mean"], stats["std"], True),
    )
    val_dataset = CachedDataset(
        config["data_root"],
        "base_val",
        build_transforms(stats["mean"], stats["std"], False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WRN16_2(num_classes=config["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    warmup = config["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["epochs"] - warmup,
            ),
        ],
        milestones=[warmup],
    )

    log_path = output_dir / f"train_log_{tag}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        writer.writerow([epoch, lr, train_loss, train_acc, val_loss, val_acc])
        log_file.flush()

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": config,
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(state, output_dir / f"best_wrn16_2_{tag}.pth")

        torch.save(state, output_dir / f"last_wrn16_2_{tag}.pth")

        print(
            f"epoch {epoch} "
            f"train_loss {train_loss:.4f} "
            f"train_acc {train_acc:.4f} "
            f"val_loss {val_loss:.4f} "
            f"val_acc {val_acc:.4f}"
        )

    log_file.close()
    elapsed = time.time() - start_time

    used_config = {
        "config": config,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }
    with open(output_dir / f"config_used_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(used_config, f, ensure_ascii=False, indent=2)

    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    epochs = [int(row["epoch"]) for row in rows]
    plt.figure()
    plt.plot(epochs, [float(row["train_loss"]) for row in rows], label="train_loss")
    plt.plot(epochs, [float(row["val_loss"]) for row in rows], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(output_dir / f"loss_curve_{tag}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [float(row["train_acc"]) for row in rows], label="train_acc")
    plt.plot(epochs, [float(row["val_acc"]) for row in rows], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(output_dir / f"acc_curve_{tag}.png", dpi=200)
    plt.close()

    print(used_config)


if __name__ == "__main__":
    main()
