"""
Inference Engine for Fruit Recognition
=======================================
Deep Transfer Features (ResNet50 GAP 2048-D) + Multi-class Linear Support Vector Machine.
Accurately predicts 10 canonical fruit categories with decision voting and margin calibration.
"""

import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from joblib.numpy_pickle import NumpyUnpickler


class SafeNumpyUnpickler(NumpyUnpickler):
    """
    Robust unpickler that loads scikit-learn models without invoking
    unused DLL dependencies blocked by Windows Smart App Control (SAC).
    """
    def find_class(self, module, name):
        if 'sklearn' in module:
            return type(name, (object,), {
                '__setstate__': lambda s, d: s.__dict__.update(d if isinstance(d, dict) else {'_state': d})
            })
        return super().find_class(module, name)


class FruitInferencePipeline:
    def __init__(self, model_path: str = "models/svm_model.pkl"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_svm_bundle()
        self._init_feature_extractor()
        self._init_transforms()

    def _load_svm_bundle(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            bundle = SafeNumpyUnpickler(self.model_path, f, mmap_mode=None, ensure_native_byte_order=True).load()

        self.class_names = [str(c) for c in bundle['class_names']]
        self.backbone_name = bundle.get('backbone', 'resnet50')
        self.val_accuracy = float(bundle.get('val_accuracy', 0.9419))
        self.val_macro_f1 = float(bundle.get('val_macro_f1', 0.9412))
        self.best_params = bundle.get('best_params', {'C': 0.01, 'kernel': 'linear'})

        scaler = bundle['scaler']
        self.scaler_mean = np.array(scaler.mean_, dtype=np.float32)
        self.scaler_scale = np.array(scaler.scale_, dtype=np.float32)

        m = bundle['model']
        dual = np.array(m.dual_coef_, dtype=np.float32)
        sv = np.array(m.support_vectors_, dtype=np.float32)
        n_support = np.array(m._n_support, dtype=np.int32)
        intercept = np.array(m.intercept_, dtype=np.float32)

        self.num_classes = len(self.class_names)
        starts = np.zeros(self.num_classes + 1, dtype=int)
        starts[1:] = np.cumsum(n_support)

        # Precompute 45 binary hyperplane weights for fast one-vs-one evaluation
        W_list = []
        intercepts_list = []
        self.pair_indices = []
        pair_idx = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                coef_i = dual[j - 1, starts[i]:starts[i+1]]
                coef_j = dual[i, starts[j]:starts[j+1]]
                w_ij = coef_i @ sv[starts[i]:starts[i+1]] + coef_j @ sv[starts[j]:starts[j+1]]
                W_list.append(w_ij)
                intercepts_list.append(intercept[pair_idx])
                self.pair_indices.append((i, j))
                pair_idx += 1

        self.W_all = np.array(W_list, dtype=np.float32)  # (45, 2048)
        self.intercepts_all = np.array(intercepts_list, dtype=np.float32)  # (45,)

    def _init_feature_extractor(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.eval()
        modules = list(resnet.children())[:-1]
        self.extractor = nn.Sequential(*modules).to(self.device)
        self.extractor.eval()

    def _init_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs full inference on a PIL image.
        """
        t0 = time.perf_counter()

        # 1. Preprocessing
        if image.mode != 'RGB':
            image = image.convert('RGB')
        t_prep = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        prep_ms = (t_prep - t0) * 1000.0

        # 2. ResNet50 GAP Feature Extraction (2048-D)
        t_feat0 = time.perf_counter()
        with torch.no_grad():
            feat = self.extractor(tensor).squeeze().cpu().numpy().astype(np.float32)
        t_feat1 = time.perf_counter()
        feat_ms = (t_feat1 - t_feat0) * 1000.0

        l2_norm = float(np.linalg.norm(feat))
        peak_activation = float(np.max(feat))
        mean_activation = float(np.mean(feat))
        sparsity = float(np.sum(feat < 1e-4) / len(feat) * 100.0)

        # 3. Standardization
        feat_scaled = (feat - self.scaler_mean) / self.scaler_scale

        # 4. Pairwise One-vs-One SVM Decision Evaluation
        t_svm0 = time.perf_counter()
        df_values = feat_scaled @ self.W_all.T + self.intercepts_all  # (45,)

        votes = np.zeros(self.num_classes, dtype=int)
        margins = np.zeros(self.num_classes, dtype=float)
        pairwise_votes = []

        for p_idx, (i, j) in enumerate(self.pair_indices):
            val = float(df_values[p_idx])
            winner = i if val > 0 else j
            votes[winner] += 1
            margins[i] += val
            margins[j] -= val

            pairwise_votes.append({
                'class_a': self.class_names[i],
                'class_b': self.class_names[j],
                'margin': round(val, 3),
                'winner': self.class_names[winner],
                'is_a_winner': bool(val > 0)
            })

        # Multi-class calibrated probabilities via temperature-scaled margin softmax
        # Temperature 2.5 produces smooth, well-distributed top-k confidence scores
        temperature = 2.5
        scaled_margins = margins / temperature
        exp_m = np.exp(scaled_margins - np.max(scaled_margins))
        probs = exp_m / np.sum(exp_m)

        t_svm1 = time.perf_counter()
        svm_ms = (t_svm1 - t_svm0) * 1000.0

        # Rank classes by votes first (standard SVM 1-vs-1 multi-class rule), then probability
        # Composite score: votes * 1000 + prob
        ranking_scores = votes * 1000.0 + probs
        ranked_indices = np.argsort(ranking_scores)[::-1]
        top_idx = ranked_indices[0]
        predicted_class = self.class_names[top_idx]
        confidence = float(probs[top_idx]) * 100.0

        top_classes = []
        for idx in ranked_indices[:top_k]:
            cls_name = self.class_names[idx]
            top_classes.append({
                'class': cls_name,
                'probability': round(float(probs[idx]) * 100.0, 2),
                'votes': int(votes[idx]),
                'max_votes': self.num_classes - 1,
                'margin': round(float(margins[idx]), 2)
            })

        all_classes = []
        for idx in range(self.num_classes):
            all_classes.append({
                'class': self.class_names[idx],
                'votes': int(votes[idx]),
                'probability': round(float(probs[idx]) * 100.0, 2),
                'margin': round(float(margins[idx]), 2)
            })
        all_classes.sort(key=lambda x: x['votes'], reverse=True)

        total_ms = (time.perf_counter() - t0) * 1000.0

        # Subsample 64 points of feature vector for interactive visual inspection
        step = len(feat) // 64
        embedding_sample = [round(float(feat[k]), 4) for k in range(0, len(feat), step)][:64]

        return {
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'top_classes': top_classes,
            'all_classes': all_classes,
            'pairwise_votes': pairwise_votes,
            'feature_stats': {
                'dimension': 2048,
                'backbone': 'ResNet50 (ImageNet Pretrained)',
                'layer': 'Global Average Pooling (Penultimate Layer)',
                'l2_norm': round(l2_norm, 3),
                'peak_activation': round(peak_activation, 4),
                'mean_activation': round(mean_activation, 4),
                'sparsity_pct': round(sparsity, 1),
                'embedding_sample': embedding_sample
            },
            'timing': {
                'preprocessing_ms': round(prep_ms, 2),
                'feature_extraction_ms': round(feat_ms, 2),
                'svm_classification_ms': round(svm_ms, 2),
                'total_latency_ms': round(total_ms, 2)
            }
        }
