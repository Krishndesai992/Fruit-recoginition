"""
Inference Script for Fruit Recognition
=====================================
Performs end-to-end inference on any new fruit image using the trained
deep-feature extraction + SVM pipeline.

Usage:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --dir path/to/test_folder/ --save_viz results/predictions
"""

import os
import sys
from pathlib import Path
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import joblib
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.data_loader import get_default_transforms, CANONICAL_CLASSES
from src.feature_extractor import DeepFeatureExtractor


def load_inference_pipeline(model_path: str = "models/svm_model.pkl"):
    """
    Loads saved model bundle and initializes feature extractor.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

    bundle = joblib.load(model_path)
    model = bundle['model']
    scaler = bundle['scaler']
    pca = bundle.get('pca', None)
    backbone_name = bundle.get('backbone', 'resnet50')
    class_names = bundle['class_names']
    input_size = bundle.get('input_size', (224, 224))

    extractor = DeepFeatureExtractor(backbone_name=backbone_name)
    transform = get_default_transforms(input_size=input_size, is_train=False, augment=False)

    return {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'class_names': class_names,
        'backbone_name': backbone_name,
        'extractor': extractor,
        'transform': transform
    }


def predict_image(pipeline: dict, image_path: str, top_k: int = 3, save_viz_path: str = None) -> dict:
    """
    Predicts fruit class for a single image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and transform image
    with Image.open(image_path) as raw_img:
        rgb_img = raw_img.convert('RGB')
        tensor = pipeline['transform'](rgb_img)

    # Extract deep feature vector
    feat_vector = pipeline['extractor'].extract_single_tensor(tensor)
    feat_vector = feat_vector.reshape(1, -1)

    # Feature scaling & optional PCA
    feat_scaled = pipeline['scaler'].transform(feat_vector)
    if pipeline['pca'] is not None:
        feat_scaled = pipeline['pca'].transform(feat_scaled)

    # Predict class & probabilities
    pred_idx = pipeline['model'].predict(feat_scaled)[0]
    pred_class = pipeline['class_names'][pred_idx]

    top_classes = []
    confidence = None

    if hasattr(pipeline['model'], 'predict_proba'):
        probs = pipeline['model'].predict_proba(feat_scaled)[0]
        sorted_indices = np.argsort(probs)[::-1]
        confidence = probs[pred_idx] * 100.0

        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            top_classes.append({
                'class': pipeline['class_names'][idx],
                'probability': float(probs[idx]) * 100.0
            })

    result = {
        'image_path': image_path,
        'predicted_class': pred_class,
        'confidence_pct': round(confidence, 2) if confidence else None,
        'top_k': top_classes
    }

    # Optional visualization
    if save_viz_path:
        os.makedirs(os.path.dirname(save_viz_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # Show input image
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Predicted: {pred_class.upper()} ({confidence:.1f}%)" if confidence else f"Predicted: {pred_class.upper()}", fontsize=12, fontweight='bold', color='darkgreen')
        axes[0].axis('off')

        # Show top probabilities bar chart
        if top_classes:
            labels = [item['class'].capitalize() for item in reversed(top_classes)]
            values = [item['probability'] for item in reversed(top_classes)]
            colors = ['#4C72B0' if i < len(labels) - 1 else '#2ca02c' for i in range(len(labels))]
            axes[1].barh(labels, values, color=colors)
            axes[1].set_xlim(0, 100)
            axes[1].set_xlabel('Probability (%)', fontsize=11)
            axes[1].set_title('Top Category Confidence', fontsize=12)
            for i, v in enumerate(values):
                axes[1].text(v + 1, i, f"{v:.1f}%", va='center', fontsize=10, fontweight='bold')
            axes[1].grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_viz_path, dpi=200)
        plt.close()
        result['viz_saved'] = save_viz_path

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fruit class for an input image")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image file")
    parser.add_argument("--dir", type=str, default=None, help="Path to a directory containing images")
    parser.add_argument("--model", type=str, default="models/svm_model.pkl", help="Path to trained model bundle")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--save_viz", type=str, default=None, help="Path to save prediction plot (or folder if --dir)")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("Please provide --image <path> or --dir <path>.")
        sys.exit(1)

    print("Loading inference pipeline...")
    pipeline = load_inference_pipeline(args.model)
    print(f"Loaded model using backbone: {pipeline['backbone_name']}")

    if args.image:
        res = predict_image(pipeline, args.image, top_k=args.top_k, save_viz_path=args.save_viz)
        print(f"\n================ PREDICTION RESULT ================")
        print(f"Image:      {res['image_path']}")
        print(f"Prediction: {res['predicted_class'].upper()}")
        if res['confidence_pct']:
            print(f"Confidence: {res['confidence_pct']:.2f}%")
        print(f"\nTop-{args.top_k} Predictions:")
        for item in res['top_k']:
            print(f"  - {item['class']:12s}: {item['probability']:5.2f}%")
        if 'viz_saved' in res:
            print(f"Visualization saved to: {res['viz_saved']}")

    elif args.dir:
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.splitext(f)[1].lower() in valid_exts]
        print(f"Found {len(images)} images in {args.dir}.\n")
        for img in images[:15]:  # Preview first 15
            viz_out = os.path.join(args.save_viz, f"pred_{os.path.basename(img)}.png") if args.save_viz else None
            res = predict_image(pipeline, img, top_k=args.top_k, save_viz_path=viz_out)
            conf_str = f"({res['confidence_pct']:.1f}%)" if res['confidence_pct'] else ""
            print(f"[{os.path.basename(img):15s}] -> {res['predicted_class']:12s} {conf_str}")
