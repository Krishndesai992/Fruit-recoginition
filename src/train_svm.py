"""
Support Vector Machine (SVM) Training & Hyperparameter Tuning
=============================================================
Trains an SVM classifier (scikit-learn SVC) on deep features extracted from
a pretrained CNN backbone. Features are scaled using StandardScaler, with
optional dimensionality reduction via PCA. Supports GridSearchCV for finding
optimal kernel (linear vs. RBF) and regularization hyperparameters (C, gamma).
"""

import os
import sys
from pathlib import Path
# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import joblib
import numpy as np
from typing import Dict, Any, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.feature_extractor import load_cached_features, extract_and_cache_dataset


def train_svm_pipeline(
    features_path: str,
    tune_hyperparameters: bool = False,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    pca_components: Optional[int] = None,
    output_model_path: str = "models/svm_model.pkl",
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and validates SVM classifier on deep feature vectors.
    """
    print(f"\n=======================================================")
    print(f"SVM Training Pipeline")
    print(f"=======================================================")
    print(f"Loading features from: {features_path}")

    data = load_cached_features(features_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    class_names = [str(c) for c in data['class_names']]
    backbone = str(data['backbone'])

    print(f"Backbone: {backbone}")
    print(f"Training samples:   {X_train.shape[0]} (Feature Dim: {X_train.shape[1]})")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    # Step 1: Feature Scaling (StandardScaler)
    print("\n[Step 1] Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Step 2: Optional PCA Dimensionality Reduction
    pca = None
    if pca_components is not None and pca_components > 0:
        print(f"\n[Step 2] Applying PCA dimensionality reduction ({pca_components} components)...")
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"Retained variance: {explained_var:.2f}% with {pca_components} components")
    else:
        print("\n[Step 2] Full deep feature dimensionality retained (PCA bypassed).")

    # Step 3: Model Training / Hyperparameter Optimization
    if tune_hyperparameters:
        print(f"\n[Step 3] Running GridSearchCV with {cv_folds}-fold Stratified CV...")
        param_grid = [
            {
                'kernel': ['linear'],
                'C': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            {
                'kernel': ['rbf'],
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        ]
        base_svc = SVC(probability=True, random_state=random_state)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            estimator=base_svc,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        best_svm = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")
        print(f"Best Hyperparameters: {best_params}")
    else:
        # Convert gamma if numeric float was passed
        try:
            parsed_gamma = float(gamma)
        except ValueError:
            parsed_gamma = gamma

        print(f"\n[Step 3] Training SVM with specified params: kernel='{kernel}', C={C}, gamma='{gamma}'...")
        best_svm = SVC(
            kernel=kernel,
            C=C,
            gamma=parsed_gamma,
            probability=True,
            random_state=random_state
        )
        best_svm.fit(X_train_scaled, y_train)
        best_params = {'kernel': kernel, 'C': C, 'gamma': parsed_gamma}

    # Step 4: Evaluate on Validation Set
    val_preds = best_svm.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"\nValidation Set Performance:")
    print(f"  Accuracy: {val_acc * 100:.2f}%")
    print(f"  Macro F1: {val_macro_f1:.4f}")

    # Step 5: Save Model Bundle
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    bundle = {
        'model': best_svm,
        'scaler': scaler,
        'pca': pca,
        'backbone': backbone,
        'class_names': class_names,
        'best_params': best_params,
        'val_accuracy': val_acc,
        'val_macro_f1': val_macro_f1,
        'input_size': (224, 224)
    }
    joblib.dump(bundle, output_model_path)
    print(f"\nSaved trained SVM model bundle to: {output_model_path}")

    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM classifier on extracted deep features")
    parser.add_argument("--features_path", type=str, default="data/processed/features_resnet50.npz", help="Path to .npz feature file")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone name if automatic path is desired")
    parser.add_argument("--tune", action="store_true", help="Perform GridSearchCV hyperparameter tuning")
    parser.add_argument("--kernel", type=str, default="rbf", choices=['linear', 'rbf'], help="SVM kernel")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter C")
    parser.add_argument("--gamma", type=str, default="scale", help="SVM gamma parameter for RBF kernel")
    parser.add_argument("--pca_components", type=int, default=None, help="Number of PCA components (optional)")
    parser.add_argument("--output_model", type=str, default="models/svm_model.pkl", help="Destination path for saved model bundle")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds for grid search")

    args = parser.parse_args()

    feat_path = args.features_path
    if args.backbone:
        feat_path = f"data/processed/features_{args.backbone}.npz"

    # If features file does not exist, extract first
    if not os.path.exists(feat_path):
        backbone_to_use = args.backbone or "resnet50"
        print(f"Features file {feat_path} not found. Running feature extractor for {backbone_to_use} first...")
        feat_path = extract_and_cache_dataset(backbone=backbone_to_use)

    train_svm_pipeline(
        features_path=feat_path,
        tune_hyperparameters=args.tune,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        pca_components=args.pca_components,
        output_model_path=args.output_model,
        cv_folds=args.cv_folds
    )
