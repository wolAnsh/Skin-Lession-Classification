import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from dataset import prepare_dataset_splits, get_transforms, ISIC2019Dataset, CLASS_NAMES
from models import build_backbone

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_backbone(
    model_name='mobilenetv3',
    epochs=100,
    batch_size=32,
    lr=0.0001,
    weight_decay=0.00002,
    label_smoothing=0.1,
    drop_rate=0.2,
    warmup_epochs=5,
    seed=42,
    data_dir='data/train-image/image',
    csv_path='data/ISIC_2019_Training_GroundTruth.csv',
    output_dir='checkpoints',
    subset_fraction=1.0,
    device=None
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Training {model_name} on device: {device}")

    # Load splits and transforms
    train_df, val_df, test_df = prepare_dataset_splits(csv_path=csv_path, img_dir=data_dir, seed=seed)
    
    if subset_fraction < 1.0:
        train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=subset_fraction, random_state=seed)).reset_index(drop=True)
        val_df = val_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=subset_fraction, random_state=seed)).reset_index(drop=True)
        print(f"Using stratified subset fraction {subset_fraction}: Train={len(train_df)}, Val={len(val_df)}")

    train_transform, eval_transform = get_transforms()
    
    train_dataset = ISIC2019Dataset(train_df, data_dir, transform=train_transform)
    val_dataset = ISIC2019Dataset(val_df, data_dir, transform=eval_transform)
    
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # Build model (unfreezing all layers as per paper)
    model = build_backbone(model_name, num_classes=len(CLASS_NAMES), pretrained=True, drop_rate=drop_rate)
    model = model.to(device)

    # Loss function with label smoothing = 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer: AdamW
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler: Linear warmup (5 epochs) followed by Cosine Annealing
    if epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pth")

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation accuracy (Section IV)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint with Val Acc: {val_acc:.4f} to {best_checkpoint_path}")

    print(f"Finished training {model_name}. Best Val Acc: {best_val_acc:.4f}")
    return best_checkpoint_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train lightweight backbone on ISIC 2019")
    parser.add_argument('--model', type=str, default='mobilenetv3', choices=['mobilenetv3', 'fasternet', 'shvit'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--subset', type=float, default=1.0, help="Fraction of dataset for quick training/testing")
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train_backbone(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_fraction=args.subset,
        output_dir=args.output_dir
    )

