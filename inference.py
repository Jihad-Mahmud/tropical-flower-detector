# inference.py
# ─────────────────────────────────────────────────────────────────────────────
# Two-stage prediction:
#   Stage 1 — ResNet50 checks if image is a flower/plant using ImageNet labels
#   Stage 2 — BYOL classifier identifies which of the 7 tropical flowers
# ─────────────────────────────────────────────────────────────────────────────

import torch
from torchvision import transforms, models
from PIL import Image

from model import FlowerClassifier, load_classifier_weights, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

CLASS_EMOJI = {
    "Bougainvillea"         : "🌺",
    "Crown of thorns"       : "🌿",
    "Hibiscus"              : "🌸",
    "Jungle geranium"       : "🌼",
    "Madagascar periwinkle" : "💜",
    "Marigold"              : "🟡",
    "Rose"                  : "🌹",
}

# ── Verified ImageNet-1K flower and plant class indices ──────────────────────
# Full list of plant/flower/nature related ImageNet classes
FLOWER_PLANT_INDICES = set([
    # Flowers
    985,  # daisy
    986,  # yellow lady's slipper (orchid type)
    984,  # rapeseed (yellow flowers)
    309,  # bee eater (often near flowers)
    # Fungus / nature
    991,  # coral fungus
    992,  # agaric
    993,  # gyromitra
    994,  # stinkhorn
    995,  # earth star
    # Fruits / plants
    949,  # strawberry
    950,  # orange
    951,  # lemon
    952,  # fig
    953,  # pineapple
    954,  # banana
    955,  # jackfruit
    956,  # cherimoya
    957,  # pomegranate
    # Vegetables / garden
    937,  # broccoli
    938,  # cauliflower
    939,  # zucchini
    940,  # spaghetti squash
    941,  # acorn squash
    942,  # butternut squash
    943,  # cucumber
    944,  # artichoke
    945,  # bell pepper
    946,  # cardoon
    947,  # mushroom
    948,  # Granny Smith apple
    # Trees / nature
    340,  # long eared owl (forests)
    998,  # coral reef
    # Additional plant classes
    920,  # greenhouse
    735,  # hare (often in grass/nature)
    # Ferns and mosses
    987,  # corn / maize plant
    988,  # acorn (from oak tree)
    989,  # hip (rose hip - direct rose relation!)
    990,  # buckeye
])

# Additional text keywords to check against ImageNet label names
FLOWER_KEYWORDS = [
    "flower", "rose", "daisy", "tulip", "orchid", "lotus",
    "hibiscus", "marigold", "geranium", "blossom", "petal",
    "bouquet", "floral", "sunflower", "dandelion", "plant",
    "shrub", "herb", "fern", "leaf", "vine", "bloom",
    "bougainvillea", "periwinkle", "begonia", "anemone",
    "rapeseed", "lady slipper", "hip",
]


def load_model(weights_path: str = "models/flower_classifier.pth"):
    """Load BYOL flower classifier + ResNet50 for flower/non-flower check."""

    # Stage 2: your trained flower classifier
    flower_model = FlowerClassifier()
    load_classifier_weights(flower_model, weights_path, DEVICE)
    flower_model.to(DEVICE)
    flower_model.eval()

    # Stage 1: ResNet50 pretrained on ImageNet
    print("[INFO] Loading ResNet50 for flower detection check...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.to(DEVICE)
    resnet.eval()

    # Load ImageNet labels for keyword matching
    imagenet_labels = _load_imagenet_labels()

    print(f"[OK] Both models ready on {DEVICE}")
    return flower_model, resnet, imagenet_labels


def _load_imagenet_labels():
    """Load ImageNet class labels for keyword-based flower detection."""
    try:
        import urllib.request, json
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        print("[INFO] Could not load ImageNet labels online — using index-only check.")
        return None


def _is_flower(resnet, imagenet_labels, tensor: torch.Tensor) -> bool:
    """
    Use ResNet50 to verify the image contains a flower or plant.
    Checks top-15 predictions against known flower/plant classes.
    """
    with torch.no_grad():
        logits = resnet(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top15  = probs.topk(15)

    top_indices = top15.indices.tolist()
    top_probs   = top15.values.tolist()

    for idx, prob in zip(top_indices, top_probs):
        # Method 1: check against known flower indices
        if idx in FLOWER_PLANT_INDICES:
            return True

        # Method 2: check label name contains flower keyword
        if imagenet_labels and idx < len(imagenet_labels):
            label = imagenet_labels[idx].lower()
            if any(kw in label for kw in FLOWER_KEYWORDS):
                return True

    return False


def predict(models_tuple, image: Image.Image) -> dict:
    flower_model, resnet, imagenet_labels = models_tuple
    tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # ── Stage 1: Is this a flower/plant? ──────────────────────────────────────
    if not _is_flower(resnet, imagenet_labels, tensor):
        return {
            "is_flower"  : False,
            "class"      : None,
            "emoji"      : None,
            "confidence" : None,
            "all_scores" : None,
        }

    # ── Stage 2: Which of the 7 tropical flowers? ─────────────────────────────
    with torch.no_grad():
        logits = flower_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_prob, top_idx = probs.max(0)
    confidence = round(top_prob.item() * 100, 2)
    top_class  = CLASS_NAMES[top_idx.item()]
    all_scores = {
        cls: round(probs[i].item() * 100, 2)
        for i, cls in enumerate(CLASS_NAMES)
    }

    return {
        "is_flower"  : True,
        "class"      : top_class,
        "emoji"      : CLASS_EMOJI.get(top_class, "🌸"),
        "confidence" : confidence,
        "all_scores" : all_scores,
    }
