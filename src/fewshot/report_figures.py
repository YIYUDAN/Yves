import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

from src.fewshot.foreground import grabcut_crop

DATA_ROOT = Path("data/fewshot/G05/images")
CLASS_DIRS = ["00", "01", "02", "03", "04"]
CLASS_NAMES = ["非洲野犬", "黑足鼬", "板条箱", "沙漏", "校车"]
OUT_DIR = Path("outputs/report")

for candidate in ("Microsoft YaHei", "SimHei"):
    if any(candidate in item.name for item in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [candidate]
        break
plt.rcParams["axes.unicode_minus"] = False


def class_files(class_dir):
    return sorted((DATA_ROOT / "support" / class_dir).glob("*.jpg"))


def query_files(class_dir):
    return sorted((DATA_ROOT / "query" / class_dir).glob("*.jpg"))


def plot_distribution():
    support_counts = [len(class_files(name)) for name in CLASS_DIRS]
    query_counts = [len(query_files(name)) for name in CLASS_DIRS]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    positions = range(len(CLASS_DIRS))
    width = 0.36
    ax.bar([p - width / 2 for p in positions], support_counts, width, label="支持集")
    ax.bar([p + width / 2 for p in positions], query_counts, width, label="查询集")
    for p, value in zip(positions, support_counts):
        ax.text(p - width / 2, value + 1.5, str(value), ha="center", fontsize=9)
    for p, value in zip(positions, query_counts):
        ax.text(p + width / 2, value + 1.5, str(value), ha="center", fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{d}\n{n}" for d, n in zip(CLASS_DIRS, CLASS_NAMES)], fontsize=9)
    ax.set_ylabel("图像数量")
    ax.set_ylim(0, 58)
    ax.legend()
    ax.set_title("G05 五类图像数量分布")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_class_distribution.png", dpi=200)
    plt.close(fig)


def plot_size():
    widths = []
    heights = []
    for name in CLASS_DIRS:
        for path in query_files(name):
            with Image.open(path) as image:
                widths.append(image.size[0])
                heights.append(image.size[1])

    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.scatter(widths, heights, s=12, alpha=0.55)
    ax.axvline(224, color="red", linewidth=1, linestyle="dashed")
    ax.axhline(224, color="red", linewidth=1, linestyle="dashed")
    ax.set_xlabel("宽度 / 像素")
    ax.set_ylabel("高度 / 像素")
    ax.set_title("查询集图像原始尺寸分布（红线为 224）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_image_size.png", dpi=200)
    plt.close(fig)


def plot_samples():
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1))
    for ax, name, zh in zip(axes, CLASS_DIRS, CLASS_NAMES):
        path = class_files(name)[0]
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{name} {zh}\n{path.name}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_samples.png", dpi=200)
    plt.close(fig)


def plot_crop_compare():
    rng = random.Random(2026)
    picks = []
    for name in CLASS_DIRS:
        files = query_files(name)
        picks.append(rng.choice(files))

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for column, path in enumerate(picks):
        with Image.open(path) as image:
            original = image.convert("RGB")
        cropped = grabcut_crop(original)
        axes[0, column].imshow(original)
        axes[0, column].set_title(f"原图 {path.name}", fontsize=8)
        axes[1, column].imshow(cropped)
        axes[1, column].set_title(f"前景裁剪 {cropped.size[0]}x{cropped.size[1]}", fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_crop_compare.png", dpi=200)
    plt.close(fig)


CASES = [
    ("01", "G05-Q-01-07.jpg", "黑足鼬", "校车", "校车"),
    ("02", "G05-Q-02-41.jpg", "板条箱", "校车", "校车"),
    ("04", "G05-Q-04-27.jpg", "校车", "沙漏", "沙漏"),
    ("01", "G05-Q-01-10.jpg", "黑足鼬", "黑足鼬", "非洲野犬"),
]


def plot_misclassified():
    fig, axes = plt.subplots(2, len(CASES), figsize=(15, 6.4))
    for column, (class_dir, name, true_name, baseline_name, adapt_name) in enumerate(CASES):
        path = DATA_ROOT / "query" / class_dir / name
        with Image.open(path) as image:
            original = image.convert("RGB")
        cropped = grabcut_crop(original)

        axes[0, column].imshow(original)
        axes[0, column].set_title(
            f"{name}\n真实 {true_name}\n方案一 {baseline_name} / 方案二 {adapt_name}",
            fontsize=9,
        )
        axes[1, column].imshow(cropped)
        axes[1, column].set_title(f"前景裁剪 {cropped.size[0]}x{cropped.size[1]}", fontsize=9)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_misclassified.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_distribution()
    plot_size()
    plot_samples()
    plot_crop_compare()
    plot_misclassified()
    print("figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
