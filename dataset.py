import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

class ISIC2019Dataset(Dataset):
    """
    ISIC 2019 Skin Lesion Dataset.
    Supports on-the-fly augmentation matching the paper:
    - Random Affine (rotation, translation, scale)
    - Random Horizontal Flip (p=0.5)
    - Random Vertical Flip (p=0.5)
    - Random Color Jitter (brightness=0.20, contrast=0.15, saturation=0.10)
    - Resize to 224x224
    - Normalization with ImageNet mean and std
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # If labels are present as columns, extract them
        if all(c in self.df.columns for c in CLASS_NAMES):
            self.labels = self.df[CLASS_NAMES].values.argmax(axis=1)
        elif 'label' in self.df.columns:
            self.labels = self.df['label'].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image']
        
        # Find image file (support .jpg, .png, etc.)
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        return image


def get_transforms():
    """
    Exact data preprocessing and augmentation pipeline as specified in Section IV-E of the paper:
    - Random Affine:
        rotation 45 to 180 degrees
        translation +- 0.125 width and height ratio
        scaling from 0.90 to 1.10
    - Horizontal Flip (p=0.5)
    - Vertical Flip (p=0.5)
    - Random Color Jitter (brightness=0.20, contrast=0.15, saturation=0.10)
    - Resize to 224x224
    - Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=(-180, 180),
            translate=(0.125, 0.125),
            scale=(0.90, 1.10)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.15,
            saturation=0.10
        ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def prepare_dataset_splits(
    csv_path='data/ISIC_2019_Training_GroundTruth.csv',
    img_dir='data/train-image/image',
    seed=42
):
    """
    Splits ISIC 2019 into 80% Train (20,265), 10% Val (2,533), and 10% Test (2,533)
    using seed=42 and stratified split across the 8 categories.
    """
    df = pd.read_csv(csv_path)
    
    # Filter out any unclassified (UNK) or verify columns
    # Determine the single ground truth label for stratification
    labels = df[CLASS_NAMES].values.argmax(axis=1)
    df['label'] = labels

    # 80% train, 20% temp (which is then split 50/50 into val and test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df['label']
    )

    print(f"Dataset split summary (seed={seed}):")
    print(f"  Train samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"  Val samples:   {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"  Test samples:  {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df

