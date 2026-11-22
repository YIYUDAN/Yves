import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = Path("outputs/pretrain/train_log.csv")
output_dir = log_path.parent

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

print("saved curves")