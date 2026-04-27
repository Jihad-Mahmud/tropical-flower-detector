# 🌸 Tropical Flower Detector
### BYOL + EfficientNet-B3 · FastAPI · 7 Flower Classes

---

## What This App Does
You upload a photo of a flower → the app tells you which of the 7 tropical
flowers it is, with a confidence percentage for every class.

**7 Supported Classes:**
Bougainvillea · Crown of thorns · Hibiscus · Jungle geranium ·
Madagascar periwinkle · Marigold · Rose

---

## Complete Step-by-Step Guide (For First-Timers)

---

### STEP 1 — Install Python
> **Where:** Your computer (one-time setup)

Make sure Python 3.9 or newer is installed.
Check by opening your terminal / command prompt and typing:
```
python --version
```
If you don't have Python, download it from https://www.python.org/downloads/

---

### STEP 2 — Set Up the Project Folder
> **Where:** Your computer's file explorer + terminal

Your final folder should look like this:
```
flower-detector/
├── model.py
├── inference.py
├── main.py
├── train_classifier.py
├── requirements.txt
├── static/
│   └── index.html
└── models/
    ├── byol_backbone.pth      ← from results.zip (byol_pipeline/byol_backbone.pth)
    └── flower_classifier.pth  ← created in Step 5 (does not exist yet)
```

**Action:** Create a folder called `flower-detector` on your Desktop (or anywhere).
Place all the .py files and the static/ folder inside it.

---

### STEP 3 — Get Your BYOL Backbone Weights
> **Where:** Your file explorer

1. Open the `results.zip` file you downloaded from Kaggle.
2. Inside the zip, find: `byol_pipeline/byol_backbone.pth`
3. Copy that file into your project's `models/` folder.

✅ Result: `flower-detector/models/byol_backbone.pth`

---

### STEP 4 — Install Requirements
> **Where:** Terminal / Command Prompt — inside the flower-detector folder

Open terminal. Navigate to your project folder:
```bash
cd Desktop/flower-detector
```

Install all required libraries:
```bash
pip install -r requirements.txt
```

Wait for it to finish (may take a few minutes the first time).

---

### STEP 5 — Prepare Your Dataset & Train the Classifier
> **Where:** Terminal — inside flower-detector folder

This step trains the small classifier head on top of your BYOL backbone.
You only need to do this ONCE.

**First, set up your dataset folder:**
```
flower-detector/
└── dataset/
    ├── Bougainvillea/        ← put Bougainvillea images here
    ├── Crown of thorns/      ← put Crown of thorns images here
    ├── Hibiscus/
    ├── Jungle geranium/
    ├── Madagascar periwinkle/
    ├── Marigold/
    └── Rose/
```
⚠️ The folder names MUST match exactly (including capital letters and spaces).
   You can reuse the same dataset you used for BYOL training.

**Then run the training script:**
```bash
python train_classifier.py
```

You will see output like:
```
Device      : cuda   (or cpu if no GPU)
Dataset dir : dataset
Total images: 4319

Epoch [01/25]  Train Loss: 0.8432  Train Acc: 72.14%  | Val Acc: 78.31%
Epoch [02/25]  Train Loss: 0.5231  Train Acc: 81.56%  | Val Acc: 83.47%  ← best
...
✅ Training complete!  Best val accuracy: 91.23%
   Saved to: models/flower_classifier.pth
```

Training takes about 5–15 minutes depending on your hardware.

---

### STEP 6 — Start the Web Server
> **Where:** Terminal — inside flower-detector folder

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
🌸 Loading flower detection model...
[OK] Classifier weights loaded from: models/flower_classifier.pth
🌸 Server ready!

INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### STEP 7 — Open the Website
> **Where:** Your web browser

Open your browser and go to:
```
http://localhost:8000
```

You will see the Flower Detector website!

1. Click the upload area or drag & drop a flower photo
2. Click "Detect Flower 🌺"
3. See the result with confidence bars for all 7 classes

---

### STEP 8 — Stop the Server
> **Where:** Terminal

Press `Ctrl + C` in the terminal to stop the server.
To start it again later, just run the Step 6 command again.

---

## Common Problems & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: models/flower_classifier.pth` | Run Step 5 first |
| `FileNotFoundError: models/byol_backbone.pth` | Copy from results.zip (Step 3) |
| Low accuracy | Make sure folder names in dataset/ match exactly |
| `Address already in use` | Change port: `--port 8001` |
| Slow on CPU | Training is slower without GPU, but still works |

---

## API Endpoints (for developers)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | The web interface |
| `/predict` | POST | Upload image → get prediction JSON |
| `/classes` | GET | List all 7 supported classes |
| `/health` | GET | Check if server is running |
| `/docs` | GET | Auto-generated API documentation |
