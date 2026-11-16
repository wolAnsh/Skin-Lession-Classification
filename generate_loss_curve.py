import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np

# Set clean IEEE-style matplotlib aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13
})

log_path = r'F:\vr project\kaggle_results\isic-2019-late-fusion-base-model.log'
with open(log_path, 'r', encoding='utf-8') as f:
    entries = json.load(f)
full_text = ''.join([e.get('data', '') for e in entries if e.get('stream_name') == 'stdout'])

pattern = re.compile(r'Epoch \[(\d+)/(\d+)\] \([^)]+\) \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)')
sections = re.split(r'Fine-Tuning Backbone: (\w+) for', full_text)

data = {}
for i in range(1, len(sections), 2):
    mname = sections[i]
    matches = pattern.findall(sections[i+1])
    epochs = [int(m[0]) for m in matches]
    train_loss = [float(m[2]) for m in matches]
    train_acc = [float(m[3]) * 100 for m in matches]
    val_loss = [float(m[4]) for m in matches]
    val_acc = [float(m[5]) * 100 for m in matches]
    data[mname] = {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }

# Create a 2x3 grid: Top Row = Loss vs Epochs, Bottom Row = Accuracy vs Epochs
fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharex=True)

models_info = [
    ('mobilenetv3', 'MobileNet-v3 Large', '#1f77b4'),
    ('fasternet', 'FasterNet-T1', '#ff7f0e'),
    ('shvit', 'shViT-s4 (Vision Transformer)', '#2ca02c')
]

for idx, (mkey, mlabel, color) in enumerate(models_info):
    ax_loss = axes[0, idx]
    ax_acc = axes[1, idx]
    mdata = data[mkey]
    ep = mdata['epochs']
    
    # --- ROW 1: LOSS CURVES ---
    ax_loss.plot(ep, mdata['train_loss'], label='Train Loss', color=color, linewidth=2.0, linestyle='-')
    ax_loss.plot(ep, mdata['val_loss'], label='Val Loss', color=color, linewidth=1.8, linestyle='--', alpha=0.9)
    min_val_idx = int(np.argmin(mdata['val_loss']))
    min_val = mdata['val_loss'][min_val_idx]
    ax_loss.scatter([ep[min_val_idx]], [min_val], color='red', s=45, zorder=5, 
                    label=f'Min Val Loss: {min_val:.3f}')
    ax_loss.set_title(f'{mlabel} (Loss)', fontweight='bold', pad=8)
    if idx == 0:
        ax_loss.set_ylabel('Loss (Label Smoothing = 0.1)')
    ax_loss.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.set_ylim(0.55, 2.55)
    
    # --- ROW 2: ACCURACY CURVES ---
    ax_acc.plot(ep, mdata['train_acc'], label='Train Acc', color=color, linewidth=2.0, linestyle='-')
    ax_acc.plot(ep, mdata['val_acc'], label='Val Acc', color=color, linewidth=1.8, linestyle='--', alpha=0.9)
    max_val_idx = int(np.argmax(mdata['val_acc']))
    max_val = mdata['val_acc'][max_val_idx]
    ax_acc.scatter([ep[max_val_idx]], [max_val], color='crimson', s=45, zorder=5, 
                   label=f'Peak Val Acc: {max_val:.2f}%')
    ax_acc.set_title(f'{mlabel} (Accuracy)', fontweight='bold', pad=8)
    ax_acc.set_xlabel('Epochs')
    if idx == 0:
        ax_acc.set_ylabel('Top-1 Accuracy (%)')
    ax_acc.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    ax_acc.grid(True, linestyle=':', alpha=0.6)
    ax_acc.set_ylim(10, 95)
    ax_acc.set_xlim(1, 25)
    ax_acc.set_xticks(range(2, 26, 4))

plt.suptitle('Training Loss and Validation Accuracy Curves Across 25 Fine-Tuning Epochs on ISIC 2019', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dirs = [
    r'F:\vr project\submission\paper_latex\figures\training_loss_curves.png',
    r'F:\vr project\submission\paper_latex\training_loss_curves.png',
    r'F:\vr project\results\training_loss_curves.png'
]

for p in out_dirs:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    plt.savefig(p, dpi=300, bbox_inches='tight')

print(f"Generated training_loss_curves.png in {len(out_dirs)} locations!")
