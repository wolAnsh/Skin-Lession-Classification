import os
import numpy as np
import pandas as pd
from dataset import prepare_dataset_splits

def preprocess_clinical_metadata(
    meta_csv='data/ISIC_2019_Training_Metadata.csv',
    ground_truth_csv='data/ISIC_2019_Training_GroundTruth.csv',
    output_dir='features',
    seed=42
):
    print("Loading clinical metadata...")
    df_meta = pd.read_csv(meta_csv)
    
    # 1. Clean Age (impute missing with median, min-max normalize)
    median_age = df_meta['age_approx'].median()
    df_meta['age_imputed'] = df_meta['age_approx'].fillna(median_age)
    min_age, max_age = df_meta['age_imputed'].min(), df_meta['age_imputed'].max()
    df_meta['age_norm'] = (df_meta['age_imputed'] - min_age) / (max_age - min_age)
    
    # 2. Clean Sex (one-hot: male, female, unknown)
    df_meta['sex_clean'] = df_meta['sex'].fillna('unknown').astype(str).str.lower()
    sex_dummies = pd.get_dummies(df_meta['sex_clean'], prefix='sex')
    
    # Ensure fixed columns
    expected_sex = ['sex_male', 'sex_female', 'sex_unknown']
    for col in expected_sex:
        if col not in sex_dummies.columns:
            sex_dummies[col] = 0
    sex_dummies = sex_dummies[expected_sex]
    
    # 3. Clean Anatomical Site (one-hot)
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
    
    # Combine tabular clinical features
    meta_features = pd.concat([df_meta[['image', 'age_norm']], sex_dummies, site_dummies], axis=1)
    print(f"Total clinical features per patient: {meta_features.shape[1] - 1} features")
    
    # Split using the exact same stratified split as the visual pipeline
    train_df, val_df, test_df = prepare_dataset_splits(csv_path=ground_truth_csv, seed=seed)
    
    train_meta_df = train_df[['image']].merge(meta_features, on='image', how='left')
    val_meta_df = val_df[['image']].merge(meta_features, on='image', how='left')
    test_meta_df = test_df[['image']].merge(meta_features, on='image', how='left')
    
    feature_cols = [c for c in meta_features.columns if c != 'image']
    
    train_meta_arr = train_meta_df[feature_cols].values.astype(np.float32)
    val_meta_arr = val_meta_df[feature_cols].values.astype(np.float32)
    test_meta_arr = test_meta_df[feature_cols].values.astype(np.float32)
    
    print(f"Metadata arrays shaped:")
    print(f"  Train: {train_meta_arr.shape}")
    print(f"  Val:   {val_meta_arr.shape}")
    print(f"  Test:  {test_meta_arr.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'train_meta.npy'), train_meta_arr)
    np.save(os.path.join(output_dir, 'val_meta.npy'), val_meta_arr)
    np.save(os.path.join(output_dir, 'test_meta.npy'), test_meta_arr)
    print("Saved train_meta.npy, val_meta.npy, test_meta.npy successfully!")

if __name__ == '__main__':
    preprocess_clinical_metadata()

