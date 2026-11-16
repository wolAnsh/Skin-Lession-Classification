import os
import argparse
import time

from train_backbone import train_backbone
from extract_features import extract_all_features
from fusion_classifiers import train_and_eval_classifiers
from visualize import run_visualizations

def run_pipeline(
    mode='all',
    epochs=100,
    batch_size=32,
    subset=1.0,
    checkpoints_dir='checkpoints',
    features_dir='features',
    results_dir='results'
):
    print("=================================================================")
    print("  LATE FEATURE FUSION FOR SKIN LESION CLASSIFICATION PIPELINE   ")
    print(f"  Mode: {mode} | Epochs: {epochs} | Batch Size: {batch_size} | Subset: {subset}")
    print("=================================================================\n")

    t_start = time.time()

    if mode in ['all', 'train']:
        print(">>> STEP 1: Fine-tuning the 3 Lightweight Backbones")
        for model_name in ['mobilenetv3', 'fasternet', 'shvit']:
            print(f"\n--- Training Backbone: {model_name} ---")
            train_backbone(
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                subset_fraction=subset,
                output_dir=checkpoints_dir
            )

    if mode in ['all', 'extract', 'eval']:
        print("\n>>> STEP 2: Feature Extraction & L2 Normalized Late Fusion")
        extract_all_features(
            mobilenet_ckpt=os.path.join(checkpoints_dir, 'mobilenetv3_best.pth'),
            fasternet_ckpt=os.path.join(checkpoints_dir, 'fasternet_best.pth'),
            shvit_ckpt=os.path.join(checkpoints_dir, 'shvit_best.pth'),
            output_dir=features_dir,
            batch_size=batch_size * 2,
            subset_fraction=subset
        )

    if mode in ['all', 'eval', 'classify']:
        print("\n>>> STEP 3: Training Level-2 Fusion Classifiers (LR, SVM, XGB, LGBM)")
        train_and_eval_classifiers(
            features_dir=features_dir,
            output_dir=results_dir
        )

        print("\n>>> STEP 4: Generating Paper Visualizations (Fig 3 & Fig 5)")
        run_visualizations(
            results_dir=results_dir,
            features_dir=features_dir
        )

    total_time = (time.time() - t_start) / 60
    print(f"\n>>> PIPELINE FINISHED IN {total_time:.2f} MINUTES <<<")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end pipeline runner")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'train', 'extract', 'eval'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--subset', type=float, default=1.0)
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--features_dir', type=str, default='features')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset=args.subset,
        checkpoints_dir=args.checkpoints_dir,
        features_dir=args.features_dir,
        results_dir=args.results_dir
    )

