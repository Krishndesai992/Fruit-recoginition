"""
Evaluation and Metrics Generation Module
========================================
Evaluates trained SVM classifier on held-out test dataset:
- Accuracy, Precision, Recall, F1-score (Macro, Weighted, and Per-Class)
- High-resolution Seaborn confusion matrix heatmap (saved to results/confusion_matrix.png)
- Comprehensive text metrics report (saved to results/metrics_report.txt)
- Multi-backbone / multi-kernel comparison benchmark (saved to results/backbone_kernel_comparison.csv)
"""

import os
import sys
from pathlib import Path
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import joblib
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from typing import Dict, Any, List

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.feature_extractor import load_cached_features


def evaluate_test_set(
    model_path: str = "models/svm_model.pkl",
    features_path: str = "data/processed/features_resnet50.npz",
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Evaluates the saved model pipeline on the held-out test split.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"SVM Test Evaluation")
    print(f"=======================================================")
    print(f"Loading model bundle from: {model_path}")
    bundle = joblib.load(model_path)

    model = bundle['model']
    scaler = bundle['scaler']
    pca = bundle.get('pca', None)
    class_names = bundle['class_names']
    backbone = bundle.get('backbone', 'resnet50')
    best_params = bundle.get('best_params', {})

    print(f"Backbone: {backbone}")
    print(f"SVM Parameters: {best_params}")

    # Load test split
    print(f"Loading test features from: {features_path}")
    data = load_cached_features(features_path)
    X_test, y_test = data['X_test'], data['y_test']

    # Preprocess features using saved scaler & PCA
    X_test_scaled = scaler.transform(X_test)
    if pca is not None:
        X_test_scaled = pca.transform(X_test_scaled)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    report_str = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n----------------- TEST SET METRICS -----------------")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Macro Precision:  {macro_prec:.4f}")
    print(f"Macro Recall:     {macro_rec:.4f}")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"Weighted F1-Score:{weighted_f1:.4f}")
    print("\nDetailed Per-Class Classification Report:\n")
    print(report_str)

    # Save Metrics Report
    report_file = os.path.join(output_dir, "metrics_report.txt")
    with open(report_file, "w") as f:
        f.write("=======================================================\n")
        f.write("Fruit Recognition: SVM on Deep Features Evaluation Report\n")
        f.write("=======================================================\n\n")
        f.write(f"Backbone:           {backbone}\n")
        f.write(f"SVM Kernel:         {best_params.get('kernel', 'unknown')}\n")
        f.write(f"SVM C:              {best_params.get('C', 'unknown')}\n")
        f.write(f"SVM Gamma:          {best_params.get('gamma', 'unknown')}\n")
        f.write(f"PCA Reduction:      {'Yes (' + str(pca.n_components_) + ' components)' if pca else 'None'}\n")
        f.write(f"Test Set Size:      {len(y_test)} images\n")
        f.write(f"Overall Accuracy:   {acc * 100:.2f}%\n")
        f.write(f"Macro Precision:    {macro_prec:.4f}\n")
        f.write(f"Macro Recall:       {macro_rec:.4f}\n")
        f.write(f"Macro F1-Score:     {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score:  {weighted_f1:.4f}\n\n")
        f.write("Per-Class Classification Report:\n")
        f.write("-------------------------------------------------------\n")
        f.write(report_str)

    print(f"Metrics report saved to: {report_file}")

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Recall Rate'}
        )
    else:
        plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Recall Rate')
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%",
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.title(f"Test Confusion Matrix — SVM on {backbone.upper()} Deep Features\nAccuracy: {acc*100:.2f}% | Macro F1: {macro_f1:.4f}", fontsize=13, pad=15)
    plt.xlabel("Predicted Fruit Class", fontsize=11, labelpad=10)
    plt.ylabel("True Fruit Class", fontsize=11, labelpad=10)
    plt.tight_layout()

    cm_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_file}")

    return {
        'accuracy': acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'backbone': backbone
    }


def run_comparative_benchmark(
    backbones: List[str] = ['resnet50', 'mobilenet_v2', 'vgg16'],
    kernels: List[str] = ['linear', 'rbf'],
    output_dir: str = "results",
    processed_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Trains and evaluates SVM models across combinations of CNN backbones
    and SVM kernels to construct a benchmark comparison table.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print("\n=======================================================")
    print("Running Multi-Backbone & Multi-Kernel Comparative Benchmark")
    print("=======================================================")

    for backbone in backbones:
        feat_path = os.path.join(processed_dir, f"features_{backbone}.npz")
        if not os.path.exists(feat_path):
            print(f"Skipping {backbone} (features not extracted at {feat_path})")
            continue

        data = load_cached_features(feat_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        dim = X_train.shape[1]

        # Fit Scaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for kernel in kernels:
            print(f"\nEvaluating: Backbone={backbone.upper()}, Kernel={kernel.upper()}...")
            c_val = 1.0
            gamma_val = 'scale' if kernel == 'rbf' else 'auto'
            svm = SVC(kernel=kernel, C=c_val, gamma=gamma_val, random_state=42)
            svm.fit(X_train_s, y_train)

            preds = svm.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec = recall_score(y_test, preds, average='macro', zero_division=0)

            results.append({
                'Backbone': backbone.upper(),
                'Feature Dim': dim,
                'SVM Kernel': kernel.upper(),
                'C': c_val,
                'Gamma': gamma_val if kernel == 'rbf' else 'N/A',
                'Test Accuracy (%)': round(acc * 100, 2),
                'Macro Precision': round(prec, 4),
                'Macro Recall': round(rec, 4),
                'Macro F1-Score': round(f1, 4)
            })

    csv_file = os.path.join(output_dir, "backbone_kernel_comparison.csv")
    if pd is not None:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        print(f"\nBenchmark results saved to: {csv_file}")
        print("\nBenchmark Summary Table:")
        print(df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df.to_string(index=False))
        # Plot comparison bar chart
        if not df.empty:
            plt.figure(figsize=(9, 5))
            sns.barplot(data=df, x='Backbone', y='Test Accuracy (%)', hue='SVM Kernel', palette='Set2')
            plt.title("Test Accuracy across Pretrained Backbones and SVM Kernels", fontsize=12)
            plt.ylim(max(0, df['Test Accuracy (%)'].min() - 5), 100)
            plt.ylabel("Test Accuracy (%)")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "backbone_comparison.png"), dpi=300)
            plt.close()
        return df
    else:
        import csv
        if results:
            fieldnames = list(results[0].keys())
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\nBenchmark results saved to: {csv_file}")
            print("\nBenchmark Summary Table:")
            for row in results:
                print(row)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SVM model on test set or run comparative benchmark")
    parser.add_argument("--model_path", type=str, default="models/svm_model.pkl", help="Path to trained SVM model bundle")
    parser.add_argument("--features_path", type=str, default="data/processed/features_resnet50.npz", help="Path to test features .npz")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--compare_all", action="store_true", help="Run benchmark comparison across backbones and kernels")

    args = parser.parse_args()

    if args.compare_all:
        run_comparative_benchmark(output_dir=args.output_dir)
    else:
        evaluate_test_set(
            model_path=args.model_path,
            features_path=args.features_path,
            output_dir=args.output_dir
        )
