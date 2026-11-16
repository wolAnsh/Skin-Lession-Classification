import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from dataset import CLASS_NAMES

def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png', title="Confusion Matrix: Fusion + Logistic Regression"):
    """
    Replicates Figure 3 of the paper:
    Seaborn heatmap with custom colormap and annotations.
    """
    plt.figure(figsize=(8, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted label', fontsize=11, fontweight='bold')
    plt.ylabel('True label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix figure to {save_path}")

def plot_tsne(features, labels, save_path='results/tsne_visualization.png', max_samples=3000, seed=42):
    """
    Replicates Figure 5 of the paper:
    2D t-SNE visualization of the fused representation across the 8 classes.
    """
    print(f"Generating t-SNE embedding (seed={seed})...")
    if len(features) > max_samples:
        indices = np.random.RandomState(seed).choice(len(features), size=max_samples, replace=False)
        features_sub = features[indices]
        labels_sub = labels[indices]
    else:
        features_sub = features
        labels_sub = labels

    perp = min(30, max(5, len(features_sub) // 4))
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=seed, init='pca', learning_rate='auto')
    embedding_2d = tsne.fit_transform(features_sub)

    plt.figure(figsize=(9, 8), dpi=300)
    palette = sns.color_palette("tab10", len(CLASS_NAMES))
    
    for idx, class_name in enumerate(CLASS_NAMES):
        mask = (labels_sub == idx)
        if np.any(mask):
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                s=12,
                alpha=0.6,
                color=palette[idx],
                label=class_name
            )

    plt.title("t-SNE visualization using Fusion (ISIC 2019)", fontsize=12, fontweight='bold', pad=12)
    plt.legend(title="Lesion Class", markerscale=2.5, loc="upper right", frameon=True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE visualization to {save_path}")

def run_visualizations(results_dir='results', features_dir='features'):
    cm_path = os.path.join(results_dir, 'confusion_matrices.npz')
    if os.path.exists(cm_path):
        cms = np.load(cm_path)
        if 'Fusion + Logistic Regression' in cms:
            cm = cms['Fusion + Logistic Regression']
            plot_confusion_matrix(cm, save_path=os.path.join(results_dir, 'fig3_confusion_matrix_lr.png'))
            
    test_fused_path = os.path.join(features_dir, 'test_fused.npz')
    if os.path.exists(test_fused_path):
        data = np.load(test_fused_path)
        plot_tsne(data['fused'], data['labels'], save_path=os.path.join(results_dir, 'fig5_tsne_fused.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate paper figures (Confusion Matrix and t-SNE)")
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--features_dir', type=str, default='features')
    args = parser.parse_args()

    run_visualizations(results_dir=args.results_dir, features_dir=args.features_dir)

