# model.py
# ─────────────────────────────────────────────────────────────────────────────
# Exact BYOL architecture from your CSE-475 Assignment 02 notebook.
# Backbone : EfficientNet-B3  →  embed_dim = 1536
# Projector: 3-layer MLP  1536 → 4096 → 256
# Predictor: 2-layer MLP   256 → 4096 → 256
#
# For inference we only need the ONLINE backbone.
# We attach a small Linear classifier head on top for flower detection.
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ── Your 7 tropical flower classes (alphabetical, as ImageFolder loads them) ─
CLASS_NAMES = [
    "Bougainvillea",
    "Crown of thorns",
    "Hibiscus",
    "Jungle geranium",
    "Madagascar periwinkle",
    "Marigold",
    "Rose",
]
NUM_CLASSES = len(CLASS_NAMES)   # 7


# ── Generic MLP with BatchNorm (copied exactly from your notebook) ─────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        dims   = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── EfficientNet-B3 backbone (same layout as your notebook) ───────────────
def _build_backbone() -> nn.Module:
    """Returns EfficientNet-B3 backbone: features + avgpool + flatten → (B, 1536)."""
    base     = models.efficientnet_b3(weights=None)
    backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    return backbone


# ── Full BYOL Online network (backbone + projector + predictor) ─────────────
class BYOLOnline(nn.Module):
    """
    Used only to load the saved byol_backbone.pth weights correctly.
    After loading, we discard projector + predictor and only keep backbone.
    """
    def __init__(self, embed_dim=1536, proj_hidden=4096, proj_out=256, pred_hidden=4096):
        super().__init__()
        self.backbone  = _build_backbone()
        self.projector = MLP(embed_dim, proj_hidden, proj_out, num_layers=3)
        self.predictor = MLP(proj_out,  pred_hidden, proj_out, num_layers=2)

    def forward(self, x):
        feat = self.backbone(x)
        z    = self.projector(feat)
        p    = self.predictor(z)
        return p, z


# ── Flower Classifier: frozen BYOL backbone + trainable head ─────────────────
class FlowerClassifier(nn.Module):
    """
    backbone  : frozen BYOL-trained EfficientNet-B3  →  1536-d features
    classifier: small MLP head  1536 → 512 → 7
    """
    def __init__(self):
        super().__init__()
        self.backbone   = _build_backbone()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():             # backbone is frozen during inference
            feat = self.backbone(x)       # (B, 1536)
        return self.classifier(feat)      # (B, 7)


# ── Weight-loading helpers ────────────────────────────────────────────────────

def load_byol_backbone_weights(
    model: FlowerClassifier,
    byol_path: str,
    device: torch.device,
) -> None:
    """
    Loads backbone weights from your BYOL byol_backbone.pth checkpoint.

    Your byol_backbone.pth (from results.zip) is saved as the backbone
    nn.Sequential directly, so keys look like:
        "0.0.0.weight"          → features[0][0] conv weight
        "0.1.0.block.0.0.weight"→ features[1][0] block conv
        etc.
    These map directly into model.backbone without any prefix changes.
    """
    if not os.path.exists(byol_path):
        print(f"[WARNING] BYOL weights not found at: {byol_path}")
        print("          Backbone will use ImageNet pretrained weights instead.")
        # Fall back to ImageNet weights — still much better than random
        base     = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
        model.backbone.load_state_dict(backbone.state_dict())
        print("[OK] ImageNet pretrained weights loaded as fallback.")
        return

    ckpt  = torch.load(byol_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)   # handle wrapped checkpoints

    # ── Strategy 1: Direct load (byol_backbone.pth saved as bare backbone) ──
    # Keys like "0.0.0.weight", "0.1.0.block.0.0.weight" etc.
    try:
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        if loaded > 100:   # sanity check — EfficientNet-B3 has ~350 tensors
            print(f"[OK] BYOL backbone loaded from: {byol_path}")
            print(f"     Tensors loaded : {loaded} / {len(state)}")
            print(f"     Missing        : {len(missing)}")
            return
    except Exception:
        pass

    # ── Strategy 2: Keys prefixed with "backbone." ──────────────────────────
    bb_state = {k[len("backbone."):]: v
                for k, v in state.items() if k.startswith("backbone.")}
    if bb_state:
        missing, unexpected = model.backbone.load_state_dict(bb_state, strict=False)
        print(f"[OK] BYOL backbone loaded (backbone.* prefix). tensors={len(bb_state)}")
        return

    # ── Strategy 3: A01-style "features.*" keys ─────────────────────────────
    feat_state = {"0." + k[len("features."):]: v
                  for k, v in state.items() if k.startswith("features.")}
    if feat_state:
        missing, unexpected = model.backbone.load_state_dict(feat_state, strict=False)
        print(f"[OK] A01 backbone weights loaded. tensors={len(feat_state)}")
        return

    # ── Fallback: ImageNet weights ───────────────────────────────────────────
    print("[WARNING] Could not map checkpoint weights — using ImageNet pretrained fallback.")
    base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    model.backbone.load_state_dict(backbone.state_dict())
    print("[OK] ImageNet pretrained weights loaded as fallback.")


def load_classifier_weights(
    model: FlowerClassifier,
    clf_path: str,
    device: torch.device,
) -> None:
    """Load the trained classifier head (+ backbone) from a saved FlowerClassifier state dict."""
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier weights not found: {clf_path}")
    state = torch.load(clf_path, map_location=device)
    model.load_state_dict(state)
    print(f"[OK] Classifier weights loaded from: {clf_path}")
