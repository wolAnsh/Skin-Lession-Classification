import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import prepare_dataset_splits, get_transforms, ISIC2019Dataset, CLASS_NAMES
from models import build_backbone

def load_backbone(model_name, checkpoint_path=None, device='cpu'):
    model = build_backbone(model_name, num_classes=len(CLASS_NAMES), pretrained=True)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for {model_name} from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"Using pretrained weights for {model_name}.")
    model = model.to(device)
    model.eval()
    return model

def extract_features_from_loader(model, loader, device, split_name=""):
    feature_list = []
    labels_list = []
    total_batches = len(loader)
    t0 = time.time()
    
    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            feats = model.extract_features(images, normalize=True)
            feature_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            
            if b_idx % 25 == 0 or b_idx == total_batches:
                elapsed = time.time() - t0
                done_samples = sum(len(x) for x in labels_list)
                speed = done_samples / (elapsed if elapsed > 0 else 1e-5)
                pct = (b_idx / total_batches) * 100
                print(f"    [{split_name}] Batch {b_idx:03d}/{total_batches:03d} ({pct:5.1f}%) | {done_samples} images | {speed:.1f} img/s")
                
    return np.concatenate(feature_list, axis=0), np.concatenate(labels_list, axis=0)

def extract_all_features(
    mobilenet_ckpt='checkpoints/mobilenetv3_best.pth',
    fasternet_ckpt='checkpoints/fasternet_best.pth',
    shvit_ckpt='checkpoints/shvit_best.pth',
    data_dir='data/train-image/image',
    csv_path='data/ISIC_2019_Training_GroundTruth.csv',
    output_dir='features',
    batch_size=64,
    subset_fraction=1.0,
    seed=42,
    device=None
):
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Feature extraction on device: {device} | Batch Size: {batch_size} | Subset: {subset_fraction}")

    # Prepare dataset splits
    train_df, val_df, test_df = prepare_dataset_splits(csv_path=csv_path, img_dir=data_dir, seed=seed)
    
    if subset_fraction < 1.0:
        train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=subset_fraction, random_state=seed)).reset_index(drop=True)
        val_df = val_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=subset_fraction, random_state=seed)).reset_index(drop=True)
        test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=subset_fraction, random_state=seed)).reset_index(drop=True)

    _, eval_transform = get_transforms()
    
    splits = {
        'train': ISIC2019Dataset(train_df, data_dir, transform=eval_transform),
        'val': ISIC2019Dataset(val_df, data_dir, transform=eval_transform),
        'test': ISIC2019Dataset(test_df, data_dir, transform=eval_transform)
    }

    models_info = [
        ('mobilenetv3', mobilenet_ckpt),
        ('fasternet', fasternet_ckpt),
        ('shvit', shvit_ckpt)
    ]

    extracted_features = {}

    for model_name, ckpt_path in models_info:
        model_cache_file = os.path.join(output_dir, f"{model_name}_features.npz")
        
        if os.path.exists(model_cache_file):
            print(f"\n--> Found cached features for {model_name} at {model_cache_file}. Loading...")
            cached = np.load(model_cache_file)
            extracted_features[model_name] = {
                'train': cached['train'],
                'val': cached['val'],
                'test': cached['test'],
                'train_labels': cached['train_labels'],
                'val_labels': cached['val_labels'],
                'test_labels': cached['test_labels']
            }
            continue

        print(f"\n--- Extracting Features: {model_name} ---")
        model = load_backbone(model_name, ckpt_path, device=device)
        model_feats = {}
        
        for split_name, dataset_obj in splits.items():
            loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)
            feats, labels = extract_features_from_loader(model, loader, device, split_name=split_name)
            model_feats[split_name] = feats
            model_feats[f"{split_name}_labels"] = labels
            print(f"  --> {split_name} done: shape {feats.shape} (L2 norm: {np.linalg.norm(feats[0]):.4f})")
            
        # Cache this model's features immediately
        np.savez_compressed(model_cache_file, **model_feats)
        print(f"Saved {model_name} features to {model_cache_file}")
        extracted_features[model_name] = model_feats
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Late Feature Fusion
    print("\n--- Performing Late Feature Fusion (Concatenation) ---")
    for split_name in ['train', 'val', 'test']:
        m1_f = extracted_features['mobilenetv3'][split_name]
        m2_f = extracted_features['fasternet'][split_name]
        m3_f = extracted_features['shvit'][split_name]
        labels = extracted_features['mobilenetv3'][f"{split_name}_labels"]
        
        fused = np.concatenate([m1_f, m2_f, m3_f], axis=1)
        save_path = os.path.join(output_dir, f"{split_name}_fused.npz")
        
        np.savez_compressed(
            save_path,
            fused=fused,
            mobilenet=m1_f,
            fasternet=m2_f,
            shvit=m3_f,
            labels=labels
        )
        print(f"Saved {split_name} fused features to {save_path}: shape {fused.shape}")

    print("\nFeature extraction and late fusion complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and fuse features from backbones")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='features')
    args = parser.parse_args()

    extract_all_features(
        batch_size=args.batch_size,
        subset_fraction=args.subset,
        output_dir=args.output_dir
    )
