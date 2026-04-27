# main.py
import os
import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from inference import load_model, predict as run_predict

# ── Auto-download model from Google Drive if not present ──────────────────────
MODEL_PATH    = "models/flower_classifier.pth"
GDRIVE_ID     = "1K6bgAfGIgwZCR4XLMGhAD3Hvjg6NCUCq"
GDRIVE_URL    = f"https://drive.google.com/uc?id={GDRIVE_ID}"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        os.makedirs("models", exist_ok=True)
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists, skipping download.")

download_model()

# ── Create the app ─────────────────────────────────────────────────────────────
app = FastAPI(title="Tropical Flower Detector 🌸")

print("\n🌸 Loading flower detection models...")
models_tuple = load_model(MODEL_PATH)
print("🌸 Server ready!\n")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WEBP images accepted.")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")
    result = run_predict(models_tuple, image)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/classes")
async def get_classes():
    from model import CLASS_NAMES
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}
