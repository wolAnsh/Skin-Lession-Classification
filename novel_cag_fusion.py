import os
import time
import random
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import CLASS_NAMES

# -------------------------------------------------------------
# 1. Cost-Sensitive Focal Loss
# -------------------------------------------------------------
class CostSensitiveFocalLoss(nn.Module):
    """
    Focal Loss with clinical penalty weights for malignant classes:
    MEL (Melanoma), BCC (Basal Cell Carcinoma), SCC (Squamous Cell Carcinoma).
    """
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
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# -------------------------------------------------------------
# 2. Novel Architecture: Cross-Attention Gated Multimodal Fusion
# -------------------------------------------------------------
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
        self.proj_mobile = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.proj_faster = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.proj_shvit = nn.Sequential(
            nn.Linear(448, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

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
        # 1. Project paradigm features to tokens
        t_mob = self.proj_mobile(f_mobile).unsqueeze(1)    # (B, 1, d)
        t_fst = self.proj_faster(f_faster).unsqueeze(1)    # (B, 1, d)
        t_shv = self.proj_shvit(f_shvit).unsqueeze(1)      # (B, 1, d)
        
        tokens = torch.cat([t_mob, t_fst, t_shv], dim=1)  # (B, 3, d)

        # 2. Cross-Attention
        if self.use_attention:
            attn_out, attn_weights = self.mha(tokens, tokens, tokens)
            tokens = self.norm1(tokens + attn_out)
            tokens = self.norm2(tokens + self.ffn(tokens))
        else:
            attn_weights = None

        # 3. Dynamic Gating vs Average Pooling
        if self.use_gating:
            flat_tokens = tokens.reshape(tokens.size(0), -1)  # (B, 3*d)
            gate_logits = self.gate_net(flat_tokens)          # (B, 3)
            gate_weights = F.softmax(gate_logits, dim=-1)     # (B, 3)
            z_visual = (tokens * gate_weights.unsqueeze(-1)).sum(dim=1) # (B, d)
        else:
            gate_weights = None
            z_visual = tokens.mean(dim=1)                     # (B, d)

        # 4. Clinical Metadata Fusion
        if self.use_metadata and x_meta is not None:
            z_meta = self.meta_net(x_meta)                    # (B, 64)
            z_final = torch.cat([z_visual, z_meta], dim=1)    # (B, d + 64)
        else:
            z_final = z_visual

        # 5. Output Logits
        logits = self.classifier(z_final)
        return logits, gate_weights, attn_weights

# -------------------------------------------------------------
# 3. Training & Evaluation Utility Functions
# -------------------------------------------------------------
def get_metrics_dict(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    # Specific clinical metrics for Melanoma (Class idx 4 in CLASS_NAMES)
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

def train_and_eval_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    epochs=40,
    lr=0.001,
    weight_decay=1e-4,
    device='cpu'
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    # Evaluate best model on test set
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
    return metrics, cm, gates_arr

# -------------------------------------------------------------
# 4. Main Novelty & Ablation Study Runner
# -------------------------------------------------------------
def run_novelty_pipeline():
    print("=================================================================")
    print("       STAGE 3: CAG-MM NOVELTY PIPELINE & ABLATION STUDY        ")
    print("=================================================================\n")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pre-extracted visual features
    train_data = np.load('features/train_fused.npz')
    val_data = np.load('features/val_fused.npz')
    test_data = np.load('features/test_fused.npz')

    # Load preprocessed clinical metadata
    train_meta = np.load('features/train_meta.npy')
    val_meta = np.load('features/val_meta.npy')
    test_meta = np.load('features/test_meta.npy')

    # Prepare PyTorch Tensors
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

    # Class counts & Clinical Cost Weights calculation
    train_labels = train_data['labels']
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    total_samples = len(train_labels)
    # Inverse frequency weights
    inv_weights = total_samples / (len(CLASS_NAMES) * class_counts.astype(np.float32))
    
    # Clinical multipliers: prioritize malignant skin cancers
    clinical_multipliers = np.ones(len(CLASS_NAMES), dtype=np.float32)
    clinical_multipliers[CLASS_NAMES.index('MEL')] = 1.6   # Melanoma (deadly)
    clinical_multipliers[CLASS_NAMES.index('BCC')] = 1.3   # Basal Cell Carcinoma
    clinical_multipliers[CLASS_NAMES.index('SCC')] = 1.4   # Squamous Cell Carcinoma
    
    focal_weights = torch.tensor(inv_weights * clinical_multipliers, dtype=torch.float32)

    # Standard Cross Entropy Loss
    criterion_ce = nn.CrossEntropyLoss()
    # Cost-Sensitive Focal Loss
    criterion_focal = CostSensitiveFocalLoss(class_weights=focal_weights, gamma=2.0)

    # -------------------------------------------------------------
    # Ablation Study Configurations
    # -------------------------------------------------------------
    ablation_experiments = [
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
            'name': '5. Proposed Full Model (CAG-MM + Focal)',
            'model': CrossAttentionGatedMultimodalFusion(use_attention=True, use_gating=True, use_metadata=True),
            'criterion': criterion_focal
        }
    ]

    ablation_results = []
    final_cm = None
    final_gates = None

    for exp in ablation_experiments:
        name = exp['name']
        print(f"\n---> Running: {name}")
        
        if exp.get('type') == 'baseline':
            # Load baseline metrics from results
            base_csv = pd.read_csv('results/classification_performance.csv')
            row = base_csv[base_csv['Model'] == 'Fusion + Logistic Regression'].iloc[0]
            cms = np.load('results/confusion_matrices.npz')
            cm_base = cms['Fusion + Logistic Regression']
            
            mel_idx = CLASS_NAMES.index('MEL')
            mel_tot = cm_base[mel_idx].sum()
            mel_sens = (cm_base[mel_idx, mel_idx] / mel_tot) * 100
            mel_miss = cm_base[mel_idx, CLASS_NAMES.index('NV')]
            
            metrics = {
                'Experiment': name,
                'Accuracy': row['Accuracy'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1-Score': row['F1-Score'],
                'Melanoma Sensitivity': mel_sens,
                'Melanoma Missed as NV': mel_miss
            }
        else:
            t0 = time.time()
            metrics, cm, gates = train_and_eval_model(
                exp['model'], train_loader, val_loader, test_loader, exp['criterion'],
                epochs=40, lr=0.001, device=device
            )
            metrics['Experiment'] = name
            print(f"  Finished in {time.time()-t0:.1f}s | Acc: {metrics['Accuracy']:.2f}% | F1: {metrics['F1-Score']:.2f}% | Mel Sensitivity: {metrics['Melanoma Sensitivity']:.1f}% (Missed as NV: {metrics['Melanoma Missed as NV']})")
            if 'Full Model' in name:
                final_cm = cm
                final_gates = gates

        ablation_results.append(metrics)

    # -------------------------------------------------------------
    # 5. Save Results & Generate Publication Figures
    # -------------------------------------------------------------
    df_ablation = pd.DataFrame(ablation_results)
    cols = ['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Melanoma Sensitivity', 'Melanoma Missed as NV']
    df_ablation = df_ablation[cols]
    
    os.makedirs('results', exist_ok=True)
    df_ablation.to_csv('results/novelty_ablation_study.csv', index=False)

    print("\n=================================================================")
    print("                  FINAL ABLATION STUDY TABLE                     ")
    print("=================================================================")
    print(df_ablation.to_string(index=False))

    # Plot Side-by-Side Confusion Matrix Comparison
    print("\nGenerating Side-by-Side Confusion Matrix Comparison...")
    cms = np.load('results/confusion_matrices.npz')
    cm_base = cms['Fusion + Logistic Regression']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0], cbar=False)
    axes[0].set_title("Base Paper: Fusion + Logistic Regression\n(129 Melanomas Missed as Nevi)", fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel("Predicted label", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("True label", fontsize=10, fontweight='bold')

    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], cbar=False)
    axes[1].set_title(f"Our Novelty: CAG-MM Fusion + Cost-Sensitive Focal Loss\n(Melanoma Missed as Nevi Reduced to {int(final_cm[CLASS_NAMES.index('MEL'), CLASS_NAMES.index('NV')])})", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel("Predicted label", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("True label", fontsize=10, fontweight='bold')

    plt.tight_layout()
    cm_comparison_path = 'results/novel_vs_base_confusion_matrix.png'
    plt.savefig(cm_comparison_path, bbox_inches='tight')
    plt.close()
    print(f"Saved side-by-side confusion matrix to {cm_comparison_path}")

    # Plot Gating Distribution (Interpretability)
    if final_gates is not None:
        print("Generating Dynamic Gating Weight Distribution Figure...")
        plt.figure(figsize=(7, 5), dpi=300)
        mean_gates = final_gates.mean(axis=0) * 100
        models_labels = ['MobileNet-v3 (Local)', 'FasterNet-T1 (Texture)', 'shViT-s4 (Global ViT)']
        bars = plt.bar(models_labels, mean_gates, color=['#3498db', '#e67e22', '#2ecc71'], width=0.5)
        plt.title("Dynamic Gating Allocation across Lightweight Backbones", fontsize=11, fontweight='bold', pad=12)
        plt.ylabel("Average Attention Contribution (%)", fontsize=10, fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.ylim(0, max(mean_gates) + 15)
        plt.tight_layout()
        gate_path = 'results/novel_dynamic_gating_weights.png'
        plt.savefig(gate_path, bbox_inches='tight')
        plt.close()
        print(f"Saved gating distribution figure to {gate_path}")

    print("\nStage 3 Novelty Pipeline finished successfully!")

if __name__ == '__main__':
    run_novelty_pipeline()

