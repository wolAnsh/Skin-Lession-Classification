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

add_md("# Late Feature Fusion of Lightweight Deep Learning Architectures for Skin Lesion Classification\n## Faithful Base Paper Implementation on Cloud GPU (NVIDIA T4 / P100)")

add_code("""# Step 1: Install required packages
!pip install timm lightgbm xgboost -q
print('Dependencies installed successfully!')
""")

add_code("""import os
import sys
import time
import math
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import timm

# Hardware & Seeds
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
    print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/features', exist_ok=True)
os.makedirs('/kaggle/working/results', exist_ok=True)
""")

add_code("""# Step 2: Auto-detect dataset paths
csv_path = None
img_dir = None

for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if 'ISIC_2019_Training_GroundTruth.csv' in f:
            csv_path = os.path.join(root, f)
            break
    if csv_path:
        break

for root, dirs, files in os.walk('/kaggle/input'):
    jpgs = [f for f in files if f.endswith('.jpg')]
    if len(jpgs) > 100:
        img_dir = root
        break

print(f'GroundTruth CSV Path: {csv_path}')
print(f'Images Directory:     {img_dir}')
assert csv_path is not None, 'Could not find GroundTruth CSV in /kaggle/input'
assert img_dir is not None, 'Could not find image folder in /kaggle/input'
""")

add_code("""# Step 3: Dataset Preparation & Stratified 80/10/10 Split
CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

df = pd.read_csv(csv_path)
print('Total dataset rows:', len(df))

# Target label calculation
labels = df[CLASS_NAMES].values.argmax(axis=1)
df['label'] = labels

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label'])

print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

class ISIC2019Dataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image']
        p = os.path.join(self.img_folder, f'{img_id}.jpg')
        if not os.path.exists(p):
            p = os.path.join(self.img_folder, f'{img_id}.png')
        image = Image.open(p).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# On-the-fly transformations matching Section IV-E of the paper
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=(-180, 180), translate=(0.125, 0.125), scale=(0.90, 1.10)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.20, contrast=0.15, saturation=0.10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ISIC2019Dataset(train_df, img_dir, transform=train_transform)
val_dataset = ISIC2019Dataset(val_df, img_dir, transform=eval_transform)
test_dataset = ISIC2019Dataset(test_df, img_dir, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
print('DataLoaders initialized.')
""")

add_code("""# Step 4: Lightweight Model Architectures
MODEL_NAMES = {
    'mobilenetv3': 'mobilenetv3_large_100',
    'fasternet': 'fasternet_t1',
    'shvit': 'shvit_s4'
}

class SkinLesionBackbone(nn.Module):
    def __init__(self, model_key, num_classes=8, drop_rate=0.2):
        super().__init__()
        self.timm_name = MODEL_NAMES[model_key]
        self.model = timm.create_model(self.timm_name, pretrained=True, num_classes=num_classes, drop_rate=drop_rate)

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x, normalize=True):
        feats_spatial = self.model.forward_features(x)
        features = self.model.forward_head(feats_spatial, pre_logits=True)
        if normalize:
            features = F.normalize(features, p=2, dim=1)
        return features

print('Model classes ready.')
""")

add_code("""# Step 5: Training Pipeline with Best-Validation Checkpointing
def train_model(model_key, epochs=25, lr=0.0001, weight_decay=0.00002):
    print(f'\\n========================================================')
    print(f'Fine-Tuning Backbone: {model_key} for {epochs} epochs on GPU')
    print(f'========================================================')
    
    model = SkinLesionBackbone(model_key, num_classes=len(CLASS_NAMES), drop_rate=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    warmup_epochs = 5
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    
    best_val_acc = 0.0
    best_ckpt_path = f'/kaggle/working/checkpoints/{model_key}_best.pth'
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == lbls).sum().item()
            train_total += lbls.size(0)
            
        scheduler.step()
        ep_loss = train_loss / train_total
        ep_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)
                
        v_loss = val_loss / val_total
        v_acc = val_correct / val_total
        elapsed = time.time() - t0
        
        print(f'Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.0f}s) | Train Loss: {ep_loss:.4f} Acc: {ep_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}')
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({'model_key': model_key, 'state_dict': model.state_dict(), 'best_acc': best_val_acc, 'epoch': epoch}, best_ckpt_path)
            print(f'  --> Saved new best checkpoint with Val Acc: {v_acc:.4f}')
            
    # Also save final checkpoint
    torch.save(model.state_dict(), f'/kaggle/working/checkpoints/{model_key}_final.pth')
    print(f'Finished {model_key}. Best Val Acc: {best_val_acc:.4f}')
    return best_ckpt_path

# Run fine-tuning for all three backbones
mobilenet_ckpt = train_model('mobilenetv3', epochs=25)
fasternet_ckpt = train_model('fasternet', epochs=25)
shvit_ckpt = train_model('shvit', epochs=25)
""")

add_code("""# Step 6: Feature Extraction & L2 Normalized Late Fusion
def extract_split_features(model, loader):
    feats_list = []
    lbls_list = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = model.extract_features(imgs, normalize=True)
            feats_list.append(feats.cpu().numpy())
            lbls_list.append(lbls.numpy())
    return np.concatenate(feats_list, axis=0), np.concatenate(lbls_list, axis=0)

splits = {'train': train_loader, 'val': val_loader, 'test': test_loader}
extracted = {s: {'models': {}} for s in splits}

for m_key in ['mobilenetv3', 'fasternet', 'shvit']:
    print(f'\\nExtracting L2-normalized features for {m_key}...')
    m = SkinLesionBackbone(m_key, num_classes=len(CLASS_NAMES)).to(device)
    ckpt = torch.load(f'/kaggle/working/checkpoints/{m_key}_best.pth', map_location=device)
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    
    for s_name, loader in splits.items():
        feats, lbls = extract_split_features(m, loader)
        extracted[s_name]['models'][m_key] = feats
        extracted[s_name]['labels'] = lbls
        print(f'  {s_name}: shape {feats.shape} (norm: {np.linalg.norm(feats[0]):.4f})')
    del m
    torch.cuda.empty_cache()

# Concatenate late feature fusion
for s_name in splits:
    m1 = extracted[s_name]['models']['mobilenetv3']
    m2 = extracted[s_name]['models']['fasternet']
    m3 = extracted[s_name]['models']['shvit']
    lbls = extracted[s_name]['labels']
    fused = np.concatenate([m1, m2, m3], axis=1)
    np.savez_compressed(
        f'/kaggle/working/features/{s_name}_fused.npz',
        fused=fused, mobilenet=m1, fasternet=m2, shvit=m3, labels=lbls
    )
    print(f'Saved {s_name} fused features: {fused.shape}')
""")

add_code("""# Step 7: Train Level-2 Classifiers (LR, SVM, LightGBM, XGBoost)
train_data = np.load('/kaggle/working/features/train_fused.npz')
val_data = np.load('/kaggle/working/features/val_fused.npz')
test_data = np.load('/kaggle/working/features/test_fused.npz')

y_train = train_data['labels']
y_val = val_data['labels']
y_test = test_data['labels']

X_train_fused = train_data['fused']
X_test_fused = test_data['fused']

results_list = []
confusion_matrices = {}

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    }

# 1. Base Models Evaluation
for name, key in [('MobileNet-v3 Large', 'mobilenet'), ('FasterNet-T1', 'fasternet'), ('shViT-s4', 'shvit')]:
    clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(train_data[key], y_train)
    t0 = time.perf_counter()
    y_pred = clf.predict(test_data[key])
    t_inf = time.perf_counter() - t0
    m = get_metrics(y_test, y_pred)
    m['Model'] = name
    m['Inference Time (s)'] = t_inf
    m['Throughput (img/s)'] = len(y_test) / t_inf
    m['Model Size (MB)'] = len(pickle.dumps(clf)) / (1024*1024)
    results_list.append(m)
    print(f'[{name}] Acc: {m["Accuracy"]:.2f}% | F1: {m["F1-Score"]:.2f}%')

# 2. Proposed Fusion Models
classifiers = {
    'Fusion + Logistic Regression': LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=SEED),
    'Fusion + SVM': SVC(C=1.0, kernel='rbf', decision_function_shape='ovo', probability=False, random_state=SEED),
    'Fusion + LightGBM': LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=SEED),
    'Fusion + XGBoost': XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, reg_lambda=1.0, tree_method='hist', n_jobs=-1, eval_metric='mlogloss', random_state=SEED)
}

for name, clf in classifiers.items():
    print(f'\\nTraining {name}...')
    clf.fit(X_train_fused, y_train)
    with open(f'/kaggle/working/results/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(clf, f)
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test_fused)
    t_inf = time.perf_counter() - t0
    m = get_metrics(y_test, y_pred)
    m['Model'] = name
    m['Inference Time (s)'] = t_inf
    m['Throughput (img/s)'] = len(y_test) / t_inf
    m['Model Size (MB)'] = len(pickle.dumps(clf)) / (1024*1024)
    results_list.append(m)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    th_rate = m['Throughput (img/s)']
    print(f'[{name}] Acc: {m["Accuracy"]:.2f}% | F1: {m["F1-Score"]:.2f}% | Speed: {th_rate:.0f} img/s')

df_res = pd.DataFrame(results_list)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Inference Time (s)', 'Throughput (img/s)', 'Model Size (MB)']]
df_res.to_csv('/kaggle/working/results/classification_performance.csv', index=False)
np.savez('/kaggle/working/results/confusion_matrices.npz', **confusion_matrices)

print('\\n========================================================')
print('FINAL PERFORMANCE TABLE (Replicating Table II & III)')
print('========================================================')
print(df_res.to_string(index=False))
""")

add_code("""# Step 8: Visualizations (Figure 3 & Figure 5)
plt.figure(figsize=(8, 7), dpi=300)
cm_lr = confusion_matrices['Fusion + Logistic Regression']
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=True)
plt.title('Figure 3: Confusion Matrix using Fusion + Logistic Regression', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Predicted label', fontsize=10, fontweight='bold')
plt.ylabel('True label', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/results/fig3_confusion_matrix_lr.png', bbox_inches='tight')
plt.close()

# t-SNE Plot (Fig 5)
idx_sub = np.random.RandomState(SEED).choice(len(X_test_fused), size=min(2500, len(X_test_fused)), replace=False)
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=SEED, init='pca', learning_rate='auto')
emb = tsne.fit_transform(X_test_fused[idx_sub])
lbls = y_test[idx_sub]

plt.figure(figsize=(9, 8), dpi=300)
palette = sns.color_palette('tab10', len(CLASS_NAMES))
for idx, cname in enumerate(CLASS_NAMES):
    mask = (lbls == idx)
    if np.any(mask):
        plt.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.65, color=palette[idx], label=cname)
plt.title('Figure 5: t-SNE visualization using Fusion + Logistic Regression', fontsize=11, fontweight='bold', pad=12)
plt.legend(title='Lesion Class', markerscale=2.5, loc='upper right')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('/kaggle/working/results/fig5_tsne_fused.png', bbox_inches='tight')
plt.close()

print('Figures generated successfully in /kaggle/working/results/!')
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

with open('kaggle_kernel/train_on_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('train_on_kaggle.ipynb written successfully!')

