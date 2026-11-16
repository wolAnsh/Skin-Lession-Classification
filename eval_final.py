import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import TSNE
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
    p = pickle.dumps(model)
    return len(p) / (1024 * 1024)

def run_evaluation():
    features_dir = 'features'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading extracted features from disk...")
    train_data = np.load(os.path.join(features_dir, 'train_fused.npz'))
    val_data = np.load(os.path.join(features_dir, 'val_fused.npz'))
    test_data = np.load(os.path.join(features_dir, 'test_fused.npz'))
    
    y_train = train_data['labels']
    y_val = val_data['labels']
    y_test = test_data['labels']
    
    X_train_fused = train_data['fused']
    X_val_fused = val_data['fused']
    X_test_fused = test_data['fused']
    
    results_list = []
    confusion_matrices = {}

    # 1. Base Models
    backbones = {
        'MobileNet-v3 Large': 'mobilenet',
        'FasterNet-T1': 'fasternet',
        'shViT-s4': 'shvit'
    }
    
    print("\nEvaluating Individual Backbones...")
    for model_name, feat_key in backbones.items():
        X_tr = train_data[feat_key]
        X_te = test_data[feat_key]
        
        clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=500, random_state=42)
        clf.fit(X_tr, y_train)
        
        t0 = time.perf_counter()
        y_pred = clf.predict(X_te)
        t_infer = time.perf_counter() - t0
        th = len(X_te) / (t_infer if t_infer > 0 else 1e-6)
        
        m = compute_metrics(y_test, y_pred)
        m['Model'] = model_name
        m['Inference Time (s)'] = t_infer
        m['Throughput (img/s)'] = th
        m['Model Size (MB)'] = get_model_size_mb(clf)
        results_list.append(m)
        print(f"  [{model_name}] Acc: {m['Accuracy']:.2f}% | F1: {m['F1-Score']:.2f}%")

    # 2. Fusion + Logistic Regression
    print("\nEvaluating Fusion + Logistic Regression...")
    lr_path = os.path.join(output_dir, 'fusion_+_logistic_regression.pkl')
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            lr_clf = pickle.load(f)
    else:
        lr_clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=42)
        lr_clf.fit(X_train_fused, y_train)
        with open(lr_path, 'wb') as f:
            pickle.dump(lr_clf, f)
            
    t0 = time.perf_counter()
    y_pred_lr = lr_clf.predict(X_test_fused)
    t_infer_lr = time.perf_counter() - t0
    th_lr = len(X_test_fused) / (t_infer_lr if t_infer_lr > 0 else 1e-6)
    
    m_lr = compute_metrics(y_test, y_pred_lr)
    m_lr['Model'] = 'Fusion + Logistic Regression'
    m_lr['Inference Time (s)'] = t_infer_lr
    m_lr['Throughput (img/s)'] = th_lr
    m_lr['Model Size (MB)'] = get_model_size_mb(lr_clf)
    results_list.append(m_lr)
    confusion_matrices['Fusion + Logistic Regression'] = confusion_matrix(y_test, y_pred_lr)
    print(f"  [Fusion + Logistic Regression] Acc: {m_lr['Accuracy']:.2f}% | F1: {m_lr['F1-Score']:.2f}% | Speed: {th_lr:.0f} img/s")

    # 3. Fusion + SVM
    print("\nEvaluating Fusion + SVM...")
    svm_path = os.path.join(output_dir, 'fusion_+_svm.pkl')
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            svm_clf = pickle.load(f)
    else:
        svm_clf = SVC(C=1.0, kernel='rbf', decision_function_shape='ovo', probability=False, random_state=42)
        svm_clf.fit(X_train_fused, y_train)
        with open(svm_path, 'wb') as f:
            pickle.dump(svm_clf, f)
            
    t0 = time.perf_counter()
    y_pred_svm = svm_clf.predict(X_test_fused)
    t_infer_svm = time.perf_counter() - t0
    th_svm = len(X_test_fused) / (t_infer_svm if t_infer_svm > 0 else 1e-6)
    
    m_svm = compute_metrics(y_test, y_pred_svm)
    m_svm['Model'] = 'Fusion + SVM'
    m_svm['Inference Time (s)'] = t_infer_svm
    m_svm['Throughput (img/s)'] = th_svm
    m_svm['Model Size (MB)'] = get_model_size_mb(svm_clf)
    results_list.append(m_svm)
    confusion_matrices['Fusion + SVM'] = confusion_matrix(y_test, y_pred_svm)
    print(f"  [Fusion + SVM] Acc: {m_svm['Accuracy']:.2f}% | F1: {m_svm['F1-Score']:.2f}%")

    # 4. Fusion + LightGBM
    print("\nEvaluating Fusion + LightGBM...")
    lgb_clf = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, reg_lambda=1.0, n_jobs=4, verbose=-1, random_state=42)
    lgb_clf.fit(X_train_fused, y_train)
    t0 = time.perf_counter()
    y_pred_lgb = lgb_clf.predict(X_test_fused)
    t_infer_lgb = time.perf_counter() - t0
    th_lgb = len(X_test_fused) / (t_infer_lgb if t_infer_lgb > 0 else 1e-6)
    
    m_lgb = compute_metrics(y_test, y_pred_lgb)
    m_lgb['Model'] = 'Fusion + LightGBM'
    m_lgb['Inference Time (s)'] = t_infer_lgb
    m_lgb['Throughput (img/s)'] = th_lgb
    m_lgb['Model Size (MB)'] = get_model_size_mb(lgb_clf)
    results_list.append(m_lgb)
    confusion_matrices['Fusion + LightGBM'] = confusion_matrix(y_test, y_pred_lgb)
    print(f"  [Fusion + LightGBM] Acc: {m_lgb['Accuracy']:.2f}% | F1: {m_lgb['F1-Score']:.2f}%")

    # 5. Fusion + XGBoost
    print("\nEvaluating Fusion + XGBoost...")
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, reg_lambda=1.0, tree_method='hist', n_jobs=4, eval_metric='mlogloss', random_state=42)
    xgb_clf.fit(X_train_fused, y_train)
    t0 = time.perf_counter()
    y_pred_xgb = xgb_clf.predict(X_test_fused)
    t_infer_xgb = time.perf_counter() - t0
    th_xgb = len(X_test_fused) / (t_infer_xgb if t_infer_xgb > 0 else 1e-6)
    
    m_xgb = compute_metrics(y_test, y_pred_xgb)
    m_xgb['Model'] = 'Fusion + XGBoost'
    m_xgb['Inference Time (s)'] = t_infer_xgb
    m_xgb['Throughput (img/s)'] = th_xgb
    m_xgb['Model Size (MB)'] = get_model_size_mb(xgb_clf)
    results_list.append(m_xgb)
    confusion_matrices['Fusion + XGBoost'] = confusion_matrix(y_test, y_pred_xgb)
    print(f"  [Fusion + XGBoost] Acc: {m_xgb['Accuracy']:.2f}% | F1: {m_xgb['F1-Score']:.2f}%")

    # Save Results
    df_results = pd.DataFrame(results_list)
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Inference Time (s)', 'Throughput (img/s)', 'Model Size (MB)']
    df_results = df_results[cols]
    df_results.to_csv(os.path.join(output_dir, 'classification_performance.csv'), index=False)
    np.savez(os.path.join(output_dir, 'confusion_matrices.npz'), **confusion_matrices)

    # Plot Confusion Matrix (Fig 3)
    print("\nGenerating Figure 3 (Confusion Matrix Heatmap)...")
    plt.figure(figsize=(8, 7), dpi=300)
    cm_lr = confusion_matrices['Fusion + Logistic Regression']
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=True)
    plt.title("Confusion Matrix using Fusion + Logistic classifier (LR) model on ISIC 2019", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Predicted label', fontsize=10, fontweight='bold')
    plt.ylabel('True label', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_confusion_matrix_lr.png'), bbox_inches='tight')
    plt.close()

    # Plot t-SNE (Fig 5)
    print("Generating Figure 5 (t-SNE Plot)...")
    max_pts = 2000
    idx_sub = np.random.RandomState(42).choice(len(X_test_fused), size=min(max_pts, len(X_test_fused)), replace=False)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(X_test_fused[idx_sub])
    lbls = y_test[idx_sub]

    plt.figure(figsize=(9, 8), dpi=300)
    palette = sns.color_palette("tab10", len(CLASS_NAMES))
    for idx, cname in enumerate(CLASS_NAMES):
        mask = (lbls == idx)
        if np.any(mask):
            plt.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.65, color=palette[idx], label=cname)
    plt.title("t-SNE visualization using Fusion + Logistic classifier (LR) model on ISIC 2019", fontsize=11, fontweight='bold', pad=12)
    plt.legend(title="Lesion Class", markerscale=2.5, loc="upper right")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_tsne_fused.png'), bbox_inches='tight')
    plt.close()

    print("\n================================================================")
    print("FINAL RESULTS TABLE")
    print("================================================================")
    print(df_results.to_string(index=False))
    print("\nAll figures and results generated successfully!")

if __name__ == '__main__':
    run_evaluation()

