import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from dataset import CLASS_NAMES

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }

def get_model_size_mb(model):
    """Computes pickled model size in MB"""
    p = pickle.dumps(model)
    return len(p) / (1024 * 1024)

def train_and_eval_classifiers(features_dir='features', output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = np.load(os.path.join(features_dir, 'train_fused.npz'))
    val_data = np.load(os.path.join(features_dir, 'val_fused.npz'))
    test_data = np.load(os.path.join(features_dir, 'test_fused.npz'))
    
    y_train = train_data['labels']
    y_val = val_data['labels']
    y_test = test_data['labels']
    
    results_list = []
    confusion_matrices = {}

    # Define classifiers with exact hyperparameters from Section V-C of the paper
    # with multi-core CPU acceleration enabled (n_jobs=-1, hist tree_method)
    classifiers = {
        'Fusion + Logistic Regression': LogisticRegression(
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'Fusion + SVM': SVC(
            C=1.0,
            kernel='rbf',
            decision_function_shape='ovo',
            probability=False,
            random_state=42
        ),
        'Fusion + XGBoost': XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            reg_lambda=1.0,
            tree_method='hist',
            early_stopping_rounds=5,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42
        ),
        'Fusion + LightGBM': LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            reg_lambda=1.0,
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )
    }

    # 1. Evaluate Individual Backbones (using Logistic Regression head on each individual feature)
    backbone_features = {
        'MobileNet-v3 Large': 'mobilenet',
        'FasterNet-T1': 'fasternet',
        'shViT-s4': 'shvit'
    }

    print("================================================================")
    print("Evaluating Individual Base Models (Pre-Fusion)")
    print("================================================================")
    for model_name, feat_key in backbone_features.items():
        X_tr = train_data[feat_key]
        X_val = val_data[feat_key]
        X_te = test_data[feat_key]
        
        clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_train)
        
        # Test evaluation
        start_t = time.perf_counter()
        y_pred = clf.predict(X_te)
        inference_time = time.perf_counter() - start_t
        throughput = len(X_te) / (inference_time if inference_time > 0 else 1e-6)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Model'] = model_name
        metrics['Inference Time (s)'] = inference_time
        metrics['Throughput (img/s)'] = throughput
        metrics['Model Size (MB)'] = get_model_size_mb(clf)
        results_list.append(metrics)
        print(f"[{model_name}] Accuracy: {metrics['Accuracy']:.2f}% | F1: {metrics['F1-Score']:.2f}%")

    # 2. Evaluate Proposed Late Feature Fusion Models
    print("\n================================================================")
    print("Evaluating Proposed Late Feature Fusion Models")
    print("================================================================")
    X_train_fused = train_data['fused']
    X_val_fused = val_data['fused']
    X_test_fused = test_data['fused']

    for name, clf in classifiers.items():
        clf_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.pkl")
        
        # If model is already trained and saved, load directly
        if os.path.exists(clf_path):
            print(f"--> Found saved model for {name} at {clf_path}. Loading...")
            try:
                with open(clf_path, 'rb') as f:
                    clf = pickle.load(f)
            except Exception as e:
                print(f"Could not load {clf_path} ({e}), retraining...")
                if 'XGBoost' in name:
                    clf.fit(X_train_fused, y_train, eval_set=[(X_val_fused, y_val)], verbose=False)
                else:
                    clf.fit(X_train_fused, y_train)
        else:
            print(f"Training {name} on {X_train_fused.shape} with {os.cpu_count()} CPU threads...")
            t0 = time.time()
            if 'XGBoost' in name:
                clf.fit(X_train_fused, y_train, eval_set=[(X_val_fused, y_val)], verbose=False)
            else:
                clf.fit(X_train_fused, y_train)
            print(f"  --> Trained in {time.time()-t0:.1f}s")
            with open(clf_path, 'wb') as f:
                pickle.dump(clf, f)
        
        # Benchmark inference on test set
        start_t = time.perf_counter()
        y_pred = clf.predict(X_test_fused)
        inference_time = time.perf_counter() - start_t
        throughput = len(X_test_fused) / (inference_time if inference_time > 0 else 1e-6)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['Model'] = name
        metrics['Inference Time (s)'] = inference_time
        metrics['Throughput (img/s)'] = throughput
        metrics['Model Size (MB)'] = get_model_size_mb(clf)
        results_list.append(metrics)
        
        # Store confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
            
        print(f"[{name}] Acc: {metrics['Accuracy']:.2f}% | F1: {metrics['F1-Score']:.2f}% | Inference: {inference_time:.3f}s ({throughput:.0f} img/s)")

    # Save summary table
    df_results = pd.DataFrame(results_list)
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Inference Time (s)', 'Throughput (img/s)', 'Model Size (MB)']
    df_results = df_results[cols]
    
    results_csv = os.path.join(output_dir, 'classification_performance.csv')
    df_results.to_csv(results_csv, index=False)
    
    # Save confusion matrices
    np.savez(os.path.join(output_dir, 'confusion_matrices.npz'), **confusion_matrices)
    
    print("\n================================================================")
    print("FINAL RESULTS TABLE (Replication of Table II & Table III)")
    print("================================================================")
    print(df_results.to_string(index=False))
    return df_results, confusion_matrices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Level-2 fusion classifiers")
    parser.add_argument('--features_dir', type=str, default='features')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    train_and_eval_classifiers(features_dir=args.features_dir, output_dir=args.output_dir)
