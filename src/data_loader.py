"""
Fruit Classification Data Loader and Preprocessing Pipeline
============================================================
Handles:
- Dataset discovery & canonical class name mapping
- Stratified 70/15/15 train / validation / test partitioning
- Image loading, resizing, and normalization for deep backbones
- PyTorch Dataset and DataLoader generation
- Optional Kaggle API download helper
"""

import os
import sys
from pathlib import Path
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Canonical 10 fruit categories mapping to fix dataset typos and case variations
CANONICAL_CLASS_MAP = {
    'apple': 'apple',
    'banana': 'banana',
    'avocado': 'avocado',
    'cherry': 'cherry',
    'kiwi': 'kiwi',
    'mango': 'mango',
    'orange': 'orange',
    'pinenapple': 'pineapple',
    'pineapple': 'pineapple',
    'strawberries': 'strawberry',
    'stawberries': 'strawberry',
    'strawberry': 'strawberry',
    'watermelon': 'watermelon'
}

CANONICAL_CLASSES = sorted(list(set(CANONICAL_CLASS_MAP.values())))
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CANONICAL_CLASSES)}

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_kaggle_dataset(target_dir: str) -> bool:
    """
    Downloads and extracts the Kaggle dataset if kaggle CLI is configured.
    Dataset: karimabdulnabi/fruit-classification10-class
    """
    os.makedirs(target_dir, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authenticated. Downloading 'karimabdulnabi/fruit-classification10-class'...")
        api.dataset_download_files('karimabdulnabi/fruit-classification10-class', path=target_dir, unzip=True)
        print(f"Dataset successfully downloaded and extracted to {target_dir}")
        return True
    except Exception as e:
        print(f"Note: Automatic Kaggle API download failed ({e}).")
        print("Please ensure kaggle.json credentials exist or manually place files in data/raw/.")
        return False


def get_default_transforms(input_size: Tuple[int, int] = (224, 224), is_train: bool = False, augment: bool = False):
    """
    Returns image transformation pipeline.
    Applies optional augmentation only for training if specified.
    """
    transform_list = [
        transforms.Resize(input_size),
    ]
    if is_train and augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transforms.Compose(transform_list)


class FruitImageDataset(Dataset):
    """
    PyTorch Dataset for fruit images.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        return img, label, path


def scan_and_collect_images(raw_dir: str) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Scans the raw directory (handling MY_data or direct folders),
    canonicalizes class names, verifies image validity, and returns
    image paths, integer labels, and class distribution counts.
    """
    # Check if MY_data subdirectory exists
    my_data_path = os.path.join(raw_dir, 'MY_data')
    search_root = my_data_path if os.path.exists(my_data_path) else raw_dir

    image_paths: List[str] = []
    labels: List[int] = []
    class_counts: Dict[str, int] = {cls: 0 for cls in CANONICAL_CLASSES}

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for root, dirs, files in os.walk(search_root):
        folder_name = os.path.basename(root).lower().strip()
        # Skip top-level 'predict' folder (unlabeled test images)
        if folder_name == 'predict' or 'predict' in root.split(os.sep):
            continue

        if folder_name in CANONICAL_CLASS_MAP:
            canonical_name = CANONICAL_CLASS_MAP[folder_name]
            label_idx = CLASS_TO_IDX[canonical_name]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, file)
                    # Simple validity check
                    if os.path.getsize(full_path) > 0:
                        image_paths.append(full_path)
                        labels.append(label_idx)
                        class_counts[canonical_name] += 1

    return image_paths, labels, class_counts


def create_train_val_test_splits(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Splits the dataset into stratified train, validation, and test subsets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Step 1: Split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_seed
    )

    # Step 2: Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=random_seed
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_data_loaders(
    raw_dir: str,
    batch_size: int = 32,
    input_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    augment_train: bool = False,
    random_seed: int = 42,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Prepares and returns PyTorch DataLoaders for train, val, and test splits.
    """
    paths, labels, counts = scan_and_collect_images(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No valid fruit images found in {raw_dir}")

    (train_p, train_l), (val_p, val_l), (test_p, test_l) = create_train_val_test_splits(
        paths, labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    train_dataset = FruitImageDataset(
        train_p, train_l,
        transform=get_default_transforms(input_size, is_train=True, augment=augment_train)
    )
    val_dataset = FruitImageDataset(
        val_p, val_l,
        transform=get_default_transforms(input_size, is_train=False, augment=False)
    )
    test_dataset = FruitImageDataset(
        test_p, test_l,
        transform=get_default_transforms(input_size, is_train=False, augment=False)
    )

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset and display splits")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset folder")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    args = parser.parse_args()

    if args.download:
        download_kaggle_dataset(args.raw_dir)

    print(f"Scanning dataset in: {args.raw_dir}")
    paths, labels, counts = scan_and_collect_images(args.raw_dir)
    print(f"Total labeled images discovered: {len(paths)}")
    print("\nClass distribution (Canonical 10 classes):")
    for cls in CANONICAL_CLASSES:
        print(f"  {cls:12s}: {counts[cls]:4d} images")

    (t_p, t_l), (v_p, v_l), (te_p, te_l) = create_train_val_test_splits(paths, labels)
    print(f"\nStratified 70/15/15 Split:")
    print(f"  Train:      {len(t_p):4d} images ({len(t_p)/len(paths)*100:.1f}%)")
    print(f"  Validation: {len(v_p):4d} images ({len(v_p)/len(paths)*100:.1f}%)")
    print(f"  Test:       {len(te_p):4d} images ({len(te_p)/len(paths)*100:.1f}%)")
