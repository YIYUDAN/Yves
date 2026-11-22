import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_split_samples(data_root, split):
    data_root = Path(data_root)

    with open(data_root / "splits.json", "r", encoding="utf-8") as f:
        split_ids = set(json.load(f)[split])

    samples = []
    with open(data_root / "manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["sample_id"] in split_ids:
                rel_path = Path(item["path"])
                samples.append((data_root / rel_path, item["label"]))

    samples.sort(key=lambda x: str(x[0]))
    return samples


def build_transforms(mean, std, is_train):
    if is_train:
        ops = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        ops = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(ops)


class PretrainDataset(Dataset):
    def __init__(self, data_root, split, transform):
        self.samples = load_split_samples(data_root, split)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label