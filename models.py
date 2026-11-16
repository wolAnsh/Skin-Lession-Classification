import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MODEL_REGISTRY = {
    'mobilenetv3': 'mobilenetv3_large_100',
    'fasternet': 'fasternet_t1',
    'shvit': 'shvit_s4'
}

FEATURE_DIMS = {
    'mobilenetv3': 1280,
    'fasternet': 1280,
    'shvit': 448
}

class SkinLesionBackbone(nn.Module):
    """
    Unified wrapper for ISIC 2019 classification backbones.
    Supports standard classification forward pass and penultimate feature extraction
    with L2 magnitude normalization as described in the paper.
    """
    def __init__(self, model_key, num_classes=8, pretrained=True, drop_rate=0.2):
        super().__init__()
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
        
        self.model_key = model_key
        self.timm_name = MODEL_REGISTRY[model_key]
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        
        # Instantiate model from timm
        self.base_model = timm.create_model(
            self.timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        self.feature_dim = FEATURE_DIMS[model_key]

    def forward(self, x, return_features=False, normalize_features=True):
        """
        Forward pass.
        If return_features=False: returns logits (batch_size, num_classes).
        If return_features=True: returns (logits, features).
        """
        feats_spatial = self.base_model.forward_features(x)
        features = self.base_model.forward_head(feats_spatial, pre_logits=True)
        
        if normalize_features:
            features = F.normalize(features, p=2, dim=1)
            
        logits = self.base_model.head.fc(features) if hasattr(self.base_model, 'head') and hasattr(self.base_model.head, 'fc') else self.base_model.forward_head(feats_spatial)
        
        if return_features:
            return logits, features
        return logits

    def extract_features(self, x, normalize=True):
        """
        Extract penultimate features without classification head.
        Normalizes vector by L2 norm if normalize=True: f / ||f||_2
        """
        feats_spatial = self.base_model.forward_features(x)
        features = self.base_model.forward_head(feats_spatial, pre_logits=True)
        if normalize:
            features = F.normalize(features, p=2, dim=1)
        return features


def build_backbone(model_name, num_classes=8, pretrained=True, drop_rate=0.2):
    return SkinLesionBackbone(
        model_key=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate
    )

