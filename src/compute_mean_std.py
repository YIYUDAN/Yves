import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.data.pretrain_dataset import load_split_samples


with open("configs/pretrain.json", "r", encoding="utf-8") as f:
    config = json.load(f)

samples = load_split_samples(config["data_root"], "base_train")

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

sum_x = torch.zeros(3)
sum_x2 = torch.zeros(3)
count = 0

for path, _ in samples:
    image = Image.open(path).convert("RGB")
    x = transform(image)
    sum_x += x.sum(dim=(1, 2))
    sum_x2 += (x * x).sum(dim=(1, 2))
    count += x.shape[1] * x.shape[2]

mean = sum_x / count
std = torch.sqrt(sum_x2 / count - mean * mean)

output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

result = {
    "mean": mean.tolist(),
    "std": std.tolist(),
}

with open(output_dir / "train_mean_std.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(result)