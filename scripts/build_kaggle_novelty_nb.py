import json

cells = []

def add_code(code_str):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code_str.strip().split('\n')]
    })

def add_md(md_str):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in md_str.strip().split('\n')]
    })

add_md("# Cross-Attention & Gated Multimodal Fusion (CAG-MM Fusion) for Skin Lesion Classification\n## Novel Architecture Implementation & Ablation Study on Kaggle GPU")

add_code("""# Step 1: Install & Imports
!pip install timm lightgbm xgboost -q

import os
import sys
import time
import math
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print('GPU Model:', torch.cuda.get_device_name(0))

os.makedirs('/kaggle/working/results', exist_ok=True)
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
""")

add_code("""# Step 2: Locate Pre-extracted Base Features & Metadata
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

train_fused_path = None
meta_csv_path = None
ground_truth_csv = None

for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'train_fused.npz':
            train_fused_path = os.path.join(root, f)
            features_dir = root
        if 'ISIC_2019_Training_Metadata.csv' in f:
            meta_csv_path = os.path.join(root, f)
        if 'ISIC_2019_Training_GroundTruth.csv' in f:
            ground_truth_csv = os.path.join(root, f)

print(f'Features directory: {features_dir}')
print(f'Metadata CSV:       {meta_csv_path}')
print(f'GroundTruth CSV:    {ground_truth_csv}')

train_data = np.load(os.path.join(features_dir, 'train_fused.npz'))
val_data   = np.load(os.path.join(features_dir, 'val_fused.npz'))
test_data  = np.load(os.path.join(features_dir, 'test_fused.npz'))

print('Visual features loaded:')
print(f'  Train: MobileNet {train_data["mobilenet"].shape}, FasterNet {train_data["fasternet"].shape}, shViT {train_data["shvit"].shape}')
print(f'  Val:   {val_data["fused"].shape}')
print(f'  Test:  {test_data["fused"].shape}')
""")

add_code("""# Step 3: Preprocess Clinical Tabular Metadata
df_meta = pd.read_csv(meta_csv_path)

# 1. Clean Age
median_age = df_meta['age_approx'].median()
df_meta['age_imputed'] = df_meta['age_approx'].fillna(median_age)
min_age, max_age = df_meta['age_imputed'].min(), df_meta['age_imputed'].max()
df_meta['age_norm'] = (df_meta['age_imputed'] - min_age) / (max_age - min_age)

# 2. Clean Sex
df_meta['sex_clean'] = df_meta['sex'].fillna('unknown').astype(str).str.lower()
sex_dummies = pd.get_dummies(df_meta['sex_clean'], prefix='sex')
for col in ['sex_male', 'sex_female', 'sex_unknown']:
    if col not in sex_dummies.columns:
        sex_dummies[col] = 0
sex_dummies = sex_dummies[['sex_male', 'sex_female', 'sex_unknown']]

# 3. Clean Anatomical Site
df_meta['site_clean'] = df_meta['anatom_site_general'].fillna('unknown').astype(str).str.lower()
site_dummies = pd.get_dummies(df_meta['site_clean'], prefix='site')
expected_sites = [
    'site_anterior torso', 'site_posterior torso', 'site_lateral torso',
    'site_head/neck', 'site_lower extremity', 'site_upper extremity',
    'site_palms/soles', 'site_oral/genital', 'site_unknown'
]
for col in expected_sites:
    if col not in site_dummies.columns:
        site_dummies[col] = 0
site_dummies = site_dummies[expected_sites]

meta_df = pd.concat([df_meta[['image', 'age_norm']], sex_dummies, site_dummies], axis=1)

# Aligned Stratified Split (Seed 42)
df_gt = pd.read_csv(ground_truth_csv)
df_gt['label'] = df_gt[CLASS_NAMES].values.argmax(axis=1)

train_df, temp_df = train_test_split(df_gt, test_size=0.20, random_state=SEED, stratify=df_gt['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label'])

meta_cols = [c for c in meta_df.columns if c != 'image']
train_meta = train_df[['image']].merge(meta_df, on='image', how='left')[meta_cols].values.astype(np.float32)
val_meta   = val_df[['image']].merge(meta_df, on='image', how='left')[meta_cols].values.astype(np.float32)
test_meta  = test_df[['image']].merge(meta_df, on='image', how='left')[meta_cols].values.astype(np.float32)

print(f'Metadata encoded: {train_meta.shape[1]} features')
print(f'  Train Meta: {train_meta.shape}')
print(f'  Test Meta:  {test_meta.shape}')
""")

add_code("""# Step 4: Define the Novel Architecture (CAG-MM Fusion)
class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            alpha = weights[targets]
            focal_loss = alpha * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class CrossAttentionGatedMultimodalFusion(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        num_classes=8,
        meta_dim=13,
        use_attention=True,
        use_gating=True,
        use_metadata=True,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        self.use_gating = use_gating
        self.use_metadata = use_metadata

        # Dimension Projections to common latent space
        self.proj_mobile = nn.Sequential(nn.Linear(1280, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_faster = nn.Sequential(nn.Linear(1280, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_shvit  = nn.Sequential(nn.Linear(448, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        # Cross-Paradigm Self-Attention
        if self.use_attention:
            self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)

        # Dynamic Gating Network
        if self.use_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 3, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )

        # Clinical Metadata Embedding
        if self.use_metadata and meta_dim > 0:
            self.meta_net = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            classifier_in_dim = embed_dim + 64
        else:
            classifier_in_dim = embed_dim

        # Final Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, f_mobile, f_faster, f_shvit, x_meta=None):
        t_mob = self.proj_mobile(f_mobile).unsqueeze(1)
        t_fst = self.proj_faster(f_faster).unsqueeze(1)
        t_shv = self.proj_shvit(f_shvit).unsqueeze(1)
        tokens = torch.cat([t_mob, t_fst, t_shv], dim=1)

        if self.use_attention:
            attn_out, attn_weights = self.mha(tokens, tokens, tokens)
            tokens = self.norm1(tokens + attn_out)
            tokens = self.norm2(tokens + self.ffn(tokens))
        else:
            attn_weights = None

        if self.use_gating:
            flat_tokens = tokens.reshape(tokens.size(0), -1)
            gate_logits = self.gate_net(flat_tokens)
            gate_weights = F.softmax(gate_logits, dim=-1)
            z_visual = (tokens * gate_weights.unsqueeze(-1)).sum(dim=1)
        else:
            gate_weights = None
            z_visual = tokens.mean(dim=1)

        if self.use_metadata and x_meta is not None:
            z_meta = self.meta_net(x_meta)
            z_final = torch.cat([z_visual, z_meta], dim=1)
        else:
            z_final = z_visual

        logits = self.classifier(z_final)
        return logits, gate_weights, attn_weights

print('Novel CAG-MM Fusion Architecture defined.')
""")

add_code("""# Step 5: Training and Evaluation Pipeline
def get_metrics_dict(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    mel_idx = CLASS_NAMES.index('MEL')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    mel_total = cm[mel_idx].sum()
    mel_correct = cm[mel_idx, mel_idx]
    mel_missed_as_nv = cm[mel_idx, CLASS_NAMES.index('NV')]
    mel_recall = (mel_correct / mel_total) * 100 if mel_total > 0 else 0.0

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Melanoma Sensitivity': mel_recall,
        'Melanoma Missed as NV': mel_missed_as_nv
    }

def train_and_eval(model, train_loader, val_loader, test_loader, criterion, epochs=40, lr=0.001):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            f_mob, f_fst, f_shv, x_meta, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits, _, _ = model(f_mob, f_fst, f_shv, x_meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                f_mob, f_fst, f_shv, x_meta, labels = [b.to(device) for b in batch]
                logits, _, _ = model(f_mob, f_fst, f_shv, x_meta)
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0) * 100
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets = [], []
    all_gate_weights = []

    with torch.no_grad():
        for batch in test_loader:
            f_mob, f_fst, f_shv, x_meta, labels = [b.to(device) for b in batch]
            logits, gates, _ = model(f_mob, f_fst, f_shv, x_meta)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            if gates is not None:
                all_gate_weights.append(gates.cpu().numpy())

    metrics = get_metrics_dict(test_targets, test_preds)
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(len(CLASS_NAMES))))
    gates_arr = np.concatenate(all_gate_weights, axis=0) if all_gate_weights else None
    return metrics, cm, gates_arr, best_state
""")

add_code("""# Step 6: Run Complete Ablation Study & Novelty Benchmarking
def make_loader(data_npz, meta_arr, batch_size=128, shuffle=False):
    f_mob = torch.tensor(data_npz['mobilenet'], dtype=torch.float32)
    f_fst = torch.tensor(data_npz['fasternet'], dtype=torch.float32)
    f_shv = torch.tensor(data_npz['shvit'], dtype=torch.float32)
    x_m   = torch.tensor(meta_arr, dtype=torch.float32)
    y     = torch.tensor(data_npz['labels'], dtype=torch.long)
    ds = TensorDataset(f_mob, f_fst, f_shv, x_m, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(train_data, train_meta, batch_size=128, shuffle=True)
val_loader   = make_loader(val_data, val_meta, batch_size=128, shuffle=False)
test_loader  = make_loader(test_data, test_meta, batch_size=128, shuffle=False)

# Class Weights for Cost-Sensitive Focal Loss
train_labels = train_data['labels']
class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
inv_weights = len(train_labels) / (len(CLASS_NAMES) * class_counts.astype(np.float32))

clinical_multipliers = np.ones(len(CLASS_NAMES), dtype=np.float32)
clinical_multipliers[CLASS_NAMES.index('MEL')] = 1.6
clinical_multipliers[CLASS_NAMES.index('BCC')] = 1.3
clinical_multipliers[CLASS_NAMES.index('SCC')] = 1.4

focal_weights = torch.tensor(inv_weights * clinical_multipliers, dtype=torch.float32)

criterion_ce = nn.CrossEntropyLoss()
criterion_focal = CostSensitiveFocalLoss(class_weights=focal_weights, gamma=2.0)

ablation_configs = [
    {
        'name': '1. Base Paper (Naive Concat + LR)',
        'type': 'baseline'
    },
    {
        'name': '2. + Cross-Paradigm Attention (CPSA)',
        'model': CrossAttentionGatedMultimodalFusion(use_attention=True, use_gating=False, use_metadata=False),
        'criterion': criterion_ce
    },
    {
        'name': '3. + Dynamic Adaptive Gating (DAG)',
        'model': CrossAttentionGatedMultimodalFusion(use_attention=True, use_gating=True, use_metadata=False),
        'criterion': criterion_ce
    },
    {
        'name': '4. + Clinical Metadata Fusion (MM)',
        'model': CrossAttentionGatedMultimodalFusion(use_attention=True, use_gating=True, use_metadata=True),
        'criterion': criterion_ce
    },
    {
        'name': '5. Proposed Full Model (CAG-MM + Focal Loss)',
        'model': CrossAttentionGatedMultimodalFusion(use_attention=True, use_gating=True, use_metadata=True),
        'criterion': criterion_focal
    }
]

ablation_results = []
final_cm = None
final_gates = None
best_model_weights = None

for exp in ablation_configs:
    name = exp['name']
    print(f'\\n---> Running Ablation: {name}')
    
    if exp.get('type') == 'baseline':
        clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=SEED)
        clf.fit(train_data['fused'], train_labels)
        y_pred = clf.predict(test_data['fused'])
        metrics = get_metrics_dict(test_data['labels'], y_pred)
        cm_base = confusion_matrix(test_data['labels'], y_pred, labels=list(range(len(CLASS_NAMES))))
    else:
        metrics, cm, gates, weights = train_and_eval(
            exp['model'], train_loader, val_loader, test_loader, exp['criterion'], epochs=45, lr=0.001
        )
        if 'Full Model' in name:
            final_cm = cm
            final_gates = gates
            best_model_weights = weights

    metrics['Architecture / Pipeline'] = name
    print(f'  Result: Acc: {metrics["Accuracy"]:.2f}% | F1: {metrics["F1-Score"]:.2f}% | Mel Sensitivity: {metrics["Melanoma Sensitivity"]:.1f}% (Missed as NV: {metrics["Melanoma Missed as NV"]})')
    ablation_results.append(metrics)

# Save best model
torch.save(best_model_weights, '/kaggle/working/checkpoints/cag_mm_novel_best.pth')

df_abl = pd.DataFrame(ablation_results)[['Architecture / Pipeline', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Melanoma Sensitivity', 'Melanoma Missed as NV']]
df_abl.to_csv('/kaggle/working/results/novelty_ablation_study.csv', index=False)

print('\\n=================================================================')
print('              FINAL ABLATION STUDY & NOVELTY RESULTS             ')
print('=================================================================')
print(df_abl.to_string(index=False))
""")

add_code("""# Step 7: Publication Visualizations (Comparative Figures)
# Figure 1: Side-by-Side Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0], cbar=False)
axes[0].set_title(f"Base Paper: Fusion + Logistic Regression\\n(Melanomas Missed as Nevus: {cm_base[CLASS_NAMES.index('MEL'), CLASS_NAMES.index('NV')]})", fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel("Predicted label", fontsize=10, fontweight='bold')
axes[0].set_ylabel("True label", fontsize=10, fontweight='bold')

sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], cbar=False)
axes[1].set_title(f"Our Novelty: CAG-MM Fusion + Cost-Sensitive Focal Loss\\n(Melanomas Missed as Nevus: {final_cm[CLASS_NAMES.index('MEL'), CLASS_NAMES.index('NV')]})", fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel("Predicted label", fontsize=10, fontweight='bold')
axes[1].set_ylabel("True label", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/kaggle/working/results/novel_vs_base_confusion_matrix.png', bbox_inches='tight')
plt.close()

# Figure 2: Dynamic Gating Allocation Distribution
plt.figure(figsize=(7, 5), dpi=300)
mean_gates = final_gates.mean(axis=0) * 100
models_labels = ['MobileNet-v3\\n(Local Edge)', 'FasterNet-T1\\n(PConv Texture)', 'shViT-s4\\n(Global Context)']
bars = plt.bar(models_labels, mean_gates, color=['#3498db', '#e67e22', '#2ecc71'], width=0.45)
plt.title("Dynamic Gating Allocation across Lightweight Backbones", fontsize=11, fontweight='bold', pad=12)
plt.ylabel("Average Attention Contribution (%)", fontsize=10, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.ylim(0, max(mean_gates) + 15)
plt.tight_layout()
plt.savefig('/kaggle/working/results/novel_dynamic_gating_weights.png', bbox_inches='tight')
plt.close()

print('Publication figures generated successfully in /kaggle/working/results/!')
""")

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbformat': 4,
            'nbformat_minor': 4
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('kaggle_novelty_kernel/train_novel_on_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('train_novel_on_kaggle.ipynb generated successfully!')

