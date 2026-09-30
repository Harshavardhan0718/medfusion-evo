# MedFusion-Evo

**Evolutionary Multimodal AI for Explainable Pneumonia Screening**

This is a hackathon-oriented research prototype with:

1. A DenseNet121 chest-X-ray branch.
2. Genetic Algorithm feature selection for structured clinical data.
3. Feature fusion with a small prediction network.
4. Grad-CAM for the X-ray branch.
5. SHAP for the clinical branch.
6. A compact static frontend that can be deployed with the FastAPI function on Vercel.

## Clinical fields used by this implementation

The paired training schema is:

`age, sex, temperature, spo2, wbc, neutrophils, lymphocytes`

These are deliberately limited to fields available in a paired chest-X-ray/clinical metadata source. Do not add fields to the inference model unless they are also present and trained on in the paired dataset.

## Important scientific point

The Genetic Algorithm runs during **training**, not as a fresh optimization for every patient. It searches over clinical feature subsets and selects the subset that scores best on the validation objective. The web app displays the selected features learned during training.

The X-ray model used by the starter is `nismal1u/chestX-rays-DenseNet121`, a public DenseNet121 pneumonia classifier. The model card describes its binary NORMAL/PNEUMONIA task and its MC-dropout uncertainty method. You should cite the model in your hackathon material.

This code is not a clinical diagnostic system. The deployed app must be presented as a research/hackathon demonstration and independently validated before any real-world use.

## Training CSV schema

Create `paired.csv` with:

```csv
patient_id,image_path,label,age,sex,temperature,spo2,wbc,neutrophils,lymphocytes
p001,data/images/a.jpg,1,56,M,38.4,93,8.5,68,23
p002,data/images/b.jpg,0,42,F,36.7,98,6.4,57,32
```

`label` is 1 for your target pneumonia class and 0 for the negative class. Make sure the label definition is consistent across the dataset.

## Train locally

From this project root:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python scripts/train_multimodal.py --csv paired.csv
```

The first run downloads the public X-ray checkpoint automatically. The script caches image embeddings in `artifacts/image_cache.npz`.

## Run locally

```bash
uvicorn api.index:app --reload
```

Open:

`http://127.0.0.1:8000/`

For a FastAPI API check:

`http://127.0.0.1:8000/docs`

In local development, the static frontend is easiest to serve with a small HTTP server from `public/` while pointing it at your local API, or use Vercel CLI for a closer production-like environment:

```bash
npm i -g vercel
vercel dev
```

## Frontend layout

`index.html`, `styles.css`, and `app.js` are kept at the project root so Vercel can serve the homepage and static assets directly. The Python backend remains under `/api`.

## Deploy to Vercel

1. Push the project root to GitHub. The root must contain `index.html`, `vercel.json`, `api/`, `ml/`, and `artifacts/`.
2. Import that GitHub repository into Vercel.
3. In Vercel, set Framework Preset to **Other** and leave the Build Command empty; Vercel can serve the root `index.html` as a static file.
4. Generate the inference artifacts with the training script, then commit the small runtime artifacts (`metadata.json`, `clinical_scaler.joblib`, `clinical_model.joblib`, `image_pca.joblib`, `fusion_model.joblib`, and `clinical_background.npy`) if you are legally allowed to redistribute them. Keep `image_cache.npz` out of Git.
5. Redeploy.

The project uses a Python FastAPI function under `/api`. Vercel's Python runtime supports FastAPI. The browser sends the compressed image to `/api/analyze`.

## Why the image is compressed in the browser

Vercel Functions have a request body limit, so the frontend resizes/compresses the X-ray before upload. Keep the resulting upload under 4 MB.

## GitHub / deployment checklist

Do not upload:

- raw clinical datasets
- patient-identifying information
- model training cache files
- real patient X-rays

Commit only model artifacts that you are legally allowed to redistribute.
