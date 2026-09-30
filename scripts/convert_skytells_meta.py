from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def make_label(value: str) -> int:
    text = str(value).strip().lower()
    if any(token in text for token in ["normal", "healthy", "none"]):
        return 0
    if any(token in text for token in ["pneumonia", "covid"]):
        return 1
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Meta.csv")
    parser.add_argument("--output", default="paired.csv")
    parser.add_argument("--image-root", default=".")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.rename(columns={
        "patientid": "patient_id",
        "filename": "image_path",
        "pO2 saturation": "spo2",
        "wbc count": "wbc",
        "neutrophil count": "neutrophils",
        "lymphocyte count": "lymphocytes",
    })
    if "finding" not in df.columns:
        raise ValueError("Meta.csv must contain a finding column.")
    df["label"] = df["finding"].map(make_label)
    df = df[df["label"] >= 0].copy()

    root = Path(args.image_root).resolve()
    df["image_path"] = df["image_path"].astype(str).map(lambda p: str((root / p).resolve()))

    cols = [
        "patient_id", "image_path", "label", "age", "sex",
        "temperature", "spo2", "wbc", "neutrophils", "lymphocytes",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df[cols].to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print("Review the mapping in make_label() before using this converter for your study.")


if __name__ == "__main__":
    main()
