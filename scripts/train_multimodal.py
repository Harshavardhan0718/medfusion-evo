from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml.image_model import ImageModelService

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLINICAL_FEATURES = [
    "age",
    "sex",
    "temperature",
    "spo2",
    "wbc",
    "neutrophils",
    "lymphocytes",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "pO2 saturation": "spo2",
        "wbc count": "wbc",
        "neutrophil count": "neutrophils",
        "lymphocyte count": "lymphocytes",
        "image": "image_path",
        "filename": "image_path",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()

    required = {"image_path", "label", *CLINICAL_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Training CSV is missing required columns: " + ", ".join(missing)
        )

    df["sex"] = df["sex"].map({"M": 1.0, "Male": 1.0, "male": 1.0, "F": 0.0, "Female": 0.0, "female": 0.0, 1: 1.0, 0: 0.0})
    for col in CLINICAL_FEATURES + ["label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["image_path", "label", *CLINICAL_FEATURES]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("label must contain both 0 and 1 classes.")
    return df


def split_rows(df: pd.DataFrame):
    if "patient_id" in df.columns:
        groups = df["patient_id"].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, val_idx = next(splitter.split(df, df["label"], groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, val_idx = next(splitter.split(df, df["label"]))
    return train_idx, val_idx


def fitness_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> float:
    if mask.sum() < 2:
        return 0.0
    cols = np.where(mask)[0]
    scaler = StandardScaler().fit(x[train_idx][:, cols])
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    model.fit(scaler.transform(x[train_idx][:, cols]), y[train_idx])
    p = model.predict_proba(scaler.transform(x[val_idx][:, cols]))[:, 1]
    try:
        return float(roc_auc_score(y[val_idx], p))
    except ValueError:
        return 0.0


def ga_select_features(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, population_size=14, generations=8):
    n_features = x.shape[1]
    population = []
    for _ in range(population_size):
        mask = np.random.rand(n_features) > 0.35
        if mask.sum() < 2:
            mask[np.random.choice(n_features, 2, replace=False)] = True
        population.append(mask)

    best = None
    for _ in range(generations):
        scored = [(fitness_mask(m, x, y, train_idx, val_idx), m.copy()) for m in population]
        scored.sort(key=lambda z: z[0], reverse=True)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]

        elite = [m for _, m in scored[: max(2, population_size // 3)]]
        next_population = [m.copy() for m in elite]
        while len(next_population) < population_size:
            a, b = random.sample(elite, 2)
            cut = random.randint(1, n_features - 1)
            child = np.concatenate([a[:cut], b[cut:]]).copy()
            mutate = np.random.rand(n_features) < 0.12
            child[mutate] = ~child[mutate]
            if child.sum() < 2:
                child[np.random.choice(n_features, 2, replace=False)] = True
            next_population.append(child)
        population = next_population

    return best[1], float(best[0])


def cache_image_embeddings(df: pd.DataFrame, cache_file: Path):
    if cache_file.exists():
        data = np.load(cache_file)
        return data["embeddings"], data["probs"]

    service = ImageModelService()
    embeddings = []
    probs = []
    for i, row in df.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        result = service.predict(image)
        embeddings.append(result["embedding"])
        probs.append(result["probability"])
        if (i + 1) % 25 == 0 or i + 1 == len(df):
            print(f"Image embeddings: {i+1}/{len(df)}")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    np.savez_compressed(cache_file, embeddings=embeddings, probs=probs)
    return embeddings, probs


def candidate_architectures():
    configs = []
    hidden1 = [32, 64, 128]
    hidden2 = [0, 16, 32]
    alpha = [1e-5, 1e-4, 1e-3]
    for h1 in hidden1:
        for h2 in hidden2:
            for a in alpha:
                layers = (h1,) if h2 == 0 else (h1, h2)
                configs.append({"hidden_layer_sizes": layers, "alpha": a})
    return configs


def search_prediction_network(x_train, y_train, x_val, y_val):
    scored = []
    for cfg in candidate_architectures():
        model = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            alpha=cfg["alpha"],
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=SEED,
            batch_size=min(64, max(8, len(x_train) // 8)),
        )
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_val)[:, 1]
        score = float(roc_auc_score(y_val, pred))
        scored.append((score, cfg))

    scored.sort(key=lambda z: z[0], reverse=True)
    return scored[0][1], scored[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Paired multimodal CSV")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--cache", default="artifacts/image_cache.npz")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(pd.read_csv(args.csv))
    print(f"Usable paired rows: {len(df)}")
    train_idx, val_idx = split_rows(df)

    x_clinical = df[CLINICAL_FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(int)

    selected_mask, ga_auc = ga_select_features(x_clinical, y, train_idx, val_idx)
    selected_features = [f for f, keep in zip(CLINICAL_FEATURES, selected_mask) if keep]
    print("GA selected clinical features:", selected_features)
    print("GA validation AUC:", ga_auc)

    clinical_scaler = StandardScaler().fit(x_clinical[train_idx][:, selected_mask])
    clinical_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clinical_model.fit(
        clinical_scaler.transform(x_clinical[train_idx][:, selected_mask]),
        y[train_idx],
    )

    embeddings, image_probs = cache_image_embeddings(df, Path(args.cache))

    # Reduce the 1024-D DenseNet embedding for the fusion network.
    pca = PCA(n_components=min(64, embeddings.shape[0] - 1, embeddings.shape[1]), random_state=SEED)
    image_train_reduced = pca.fit_transform(embeddings[train_idx]).astype(np.float32)
    image_val_reduced = pca.transform(embeddings[val_idx]).astype(np.float32)
    clinical_train = clinical_scaler.transform(x_clinical[train_idx][:, selected_mask]).astype(np.float32)
    clinical_val = clinical_scaler.transform(x_clinical[val_idx][:, selected_mask]).astype(np.float32)

    fused_train = np.concatenate([image_train_reduced, clinical_train], axis=1)
    fused_val = np.concatenate([image_val_reduced, clinical_val], axis=1)

    best_arch, fusion_auc = search_prediction_network(fused_train, y[train_idx], fused_val, y[val_idx])
    print("Best prediction network:", best_arch)
    print("Fusion validation AUC:", fusion_auc)

    fusion_model = MLPClassifier(
        hidden_layer_sizes=best_arch["hidden_layer_sizes"],
        alpha=best_arch["alpha"],
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
        batch_size=min(64, max(8, len(fused_train) // 8)),
    )
    # Refit on all available paired data after architecture selection.
    full_clinical_scaled = clinical_scaler.transform(x_clinical[:, selected_mask]).astype(np.float32)
    full_image_reduced = pca.transform(embeddings).astype(np.float32)
    fused_full = np.concatenate([full_image_reduced, full_clinical_scaled], axis=1)
    fusion_model.fit(fused_full, y)

    final_fused_val = fusion_model.predict_proba(fused_val)[:, 1]
    final_auc = float(roc_auc_score(y[val_idx], final_fused_val))
    final_acc = float(accuracy_score(y[val_idx], (final_fused_val >= 0.5).astype(int)))

    joblib.dump(clinical_scaler, artifact_dir / "clinical_scaler.joblib")
    joblib.dump(clinical_model, artifact_dir / "clinical_model.joblib")
    joblib.dump(pca, artifact_dir / "image_pca.joblib")
    joblib.dump(fusion_model, artifact_dir / "fusion_model.joblib")

    # Save scaled training background for SHAP. Use only selected clinical fields.
    np.save(artifact_dir / "clinical_background.npy", full_clinical_scaled[: min(100, len(full_clinical_scaled))])

    metadata = {
        "target_name": "Pneumonia",
        "clinical_features": CLINICAL_FEATURES,
        "selected_clinical_features": selected_features,
        "ga_feature_selection_validation_auc": ga_auc,
        "fusion_validation_auc": final_auc,
        "fusion_validation_accuracy": final_acc,
        "fusion_architecture": str(best_arch["hidden_layer_sizes"]),
        "fusion_alpha": best_arch["alpha"],
        "image_embedding_dim": int(embeddings.shape[1]),
        "image_reduced_dim": int(pca.n_components_),
        "image_model_repo": "nismal1u/chestX-rays-DenseNet121",
        "note": "Research/hackathon prototype. External image model and target-data distribution require independent validation before clinical use.",
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("Training complete. Deploy the generated artifacts together with the app.")


if __name__ == "__main__":
    main()
