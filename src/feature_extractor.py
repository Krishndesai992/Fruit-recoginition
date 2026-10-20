"""
Deep Feature Extraction Module
==============================
Extracts deep features from fruit images using pretrained CNN backbones
(ResNet50, MobileNetV2, VGG16) with ImageNet weights.
No fine-tuning of the CNN is performed (transfer learning feature extractor).
The top classification layer is removed, yielding dense representation vectors.
"""

import os
import sys
from pathlib import Path
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.hub
import torchvision.models as models
from tqdm import tqdm

from src.data_loader import (
    get_data_loaders,
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_default_transforms
)


# Robust download patch to avoid Windows urllib WinError 10054 handshake issues
def _safe_download_url_to_file(url, dst, hash_prefix=None, progress=True):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    temp_dst = dst + '.tmp'
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(temp_dst, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dst), disable=not progress) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(temp_dst, dst)

torch.hub.download_url_to_file = _safe_download_url_to_file


class DeepFeatureExtractor:
    """
    Wraps a pretrained CNN backbone, strips its classification head,
    and extracts deep embedding vectors via Global Average Pooling.
    """
    SUPPORTED_BACKBONES = ['resnet50', 'mobilenet_v2', 'vgg16']

    def __init__(self, backbone_name: str = 'resnet50', device: Optional[torch.device] = None):
        self.backbone_name = backbone_name.lower().strip()
        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: '{self.backbone_name}'. Choose from: {self.SUPPORTED_BACKBONES}")

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model, self.feature_dim = self._build_feature_extractor()
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_feature_extractor(self) -> Tuple[nn.Module, int]:
        """
        Loads the backbone with pretrained ImageNet weights, removes
        the final linear classification layer, and appends GAP if necessary.
        """
        if self.backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
            # ResNet50 penultimate layer output before fc: shape (N, 2048, 1, 1) -> flatten to (N, 2048)
            feature_extractor = nn.Sequential(
                *list(base_model.children())[:-1],
                nn.Flatten()
            )
            feature_dim = 2048

        elif self.backbone_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT
            base_model = models.mobilenet_v2(weights=weights)
            # MobileNetV2 features + adaptive pool: shape (N, 1280)
            feature_extractor = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 1280

        elif self.backbone_name == 'vgg16':
            weights = models.VGG16_Weights.DEFAULT
            base_model = models.vgg16(weights=weights)
            # VGG16 features (512 channels) + adaptive pool to (1, 1) -> GAP (512-dim)
            # This is cleaner, faster, and prevents overfitting compared to 4096-dim dense FC
            feature_extractor = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 512

        else:
            raise NotImplementedError(f"Backbone {self.backbone_name} not implemented")

        return feature_extractor, feature_dim

    def extract_from_loader(self, loader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts feature vectors, labels, and file paths for all batches in a DataLoader.
        """
        features_list = []
        labels_list = []
        paths_list = []

        with torch.no_grad():
            for batch_images, batch_labels, batch_paths in tqdm(loader, desc=f"Extracting features [{self.backbone_name}]"):
                batch_images = batch_images.to(self.device)
                feats = self.model(batch_images)
                features_list.append(feats.cpu().numpy())
                labels_list.append(batch_labels.numpy())
                paths_list.extend(batch_paths)

        X = np.vstack(features_list)
        y = np.concatenate(labels_list)
        return X, y, paths_list

    def extract_single_tensor(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts feature vector for a single preprocessed image tensor (1, C, H, W).
        """
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            feat = self.model(img_tensor)
        return feat.cpu().numpy().squeeze(0)


def extract_and_cache_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    backbone: str = "resnet50",
    batch_size: int = 64,
    random_seed: int = 42
) -> str:
    """
    Extracts deep features for train, val, and test splits and saves them
    into a compressed .npz archive in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, f"features_{backbone}.npz")

    print(f"\n=======================================================")
    print(f"Deep Feature Extraction: {backbone.upper()}")
    print(f"=======================================================")

    loaders = get_data_loaders(
        raw_dir=raw_dir,
        batch_size=batch_size,
        input_size=(224, 224),
        random_seed=random_seed
    )

    extractor = DeepFeatureExtractor(backbone_name=backbone)
    print(f"Backbone: {backbone} | Feature dimension: {extractor.feature_dim} | Device: {extractor.device}")

    print("\n[1/3] Extracting training split features...")
    X_train, y_train, paths_train = extractor.extract_from_loader(loaders['train'])

    print("[2/3] Extracting validation split features...")
    X_val, y_val, paths_val = extractor.extract_from_loader(loaders['val'])

    print("[3/3] Extracting test split features...")
    X_test, y_test, paths_test = extractor.extract_from_loader(loaders['test'])

    print(f"\nExtracted shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    np.savez_compressed(
        cache_path,
        X_train=X_train, y_train=y_train, paths_train=np.array(paths_train),
        X_val=X_val, y_val=y_val, paths_val=np.array(paths_val),
        X_test=X_test, y_test=y_test, paths_test=np.array(paths_test),
        class_names=np.array(CANONICAL_CLASSES),
        backbone=backbone,
        feature_dim=extractor.feature_dim
    )
    print(f"Successfully cached extracted features to: {cache_path}")
    return cache_path


def load_cached_features(cache_path: str) -> Dict[str, np.ndarray]:
    """
    Loads pre-extracted deep features from a .npz file.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached features file not found: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract deep features using pretrained CNN backbones")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset folder")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save extracted features")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=['resnet50', 'mobilenet_v2', 'vgg16'], help="CNN Backbone")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--all_backbones", action="store_true", help="Extract features for all three supported backbones")
    args = parser.parse_args()

    if args.all_backbones:
        for b in DeepFeatureExtractor.SUPPORTED_BACKBONES:
            extract_and_cache_dataset(args.raw_dir, args.output_dir, backbone=b, batch_size=args.batch_size)
    else:
        extract_and_cache_dataset(args.raw_dir, args.output_dir, backbone=args.backbone, batch_size=args.batch_size)
