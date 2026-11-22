import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.pretrain_dataset import PretrainDataset, build_transforms
from src.models.wrn import WRN16_2


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += torch.eq(pred, labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, labels).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total


def main():
    with open("configs/pretrain.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = True

    with open(output_dir / "train_mean_std.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    train_dataset = PretrainDataset(
        config["data_root"],
        "base_train",
        build_transforms(stats["mean"], stats["std"], True),
    )
    val_dataset = PretrainDataset(
        config["data_root"],
        "base_val",
        build_transforms(stats["mean"], stats["std"], False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WRN16_2(num_classes=config["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )

    if device.type.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    log_path = output_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(
        [
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
        ]
    )

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
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()

        writer.writerow(
            [
                epoch,
                lr,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            ]
        )
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
            torch.save(state, output_dir / "best_wrn16_2.pth")

        torch.save(state, output_dir / "last_wrn16_2.pth")

        print(
            f"epoch {epoch} "
            f"train_loss {train_loss:.4f} "
            f"train_acc {train_acc:.4f} "
            f"val_loss {val_loss:.4f} "
            f"val_acc {val_acc:.4f}"
        )

    log_file.close()
    elapsed = time.time() - start_time

    max_memory_mb = 0.0
    if device.type.startswith("cuda"):
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    used_config = {
        "config": config,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "max_memory_mb": max_memory_mb,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }

    with open(output_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(used_config, f, ensure_ascii=False, indent=2)

    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(output_dir / "acc_curve.png", dpi=200)
    plt.close()

    print(used_config)


if __name__ == "__main__":
    main()