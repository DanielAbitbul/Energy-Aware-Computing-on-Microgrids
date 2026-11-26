from pathlib import Path

import pandas as pd

repository_root = Path(__file__).resolve().parents[1]
files = [
    repository_root / "data/raw/train.parquet",
    repository_root / "data/raw/val.parquet",
    repository_root / "data/raw/test.parquet",
]

for path in files:
    df = pd.read_parquet(path)

    print(f"\n{'=' * 60}")
    print(f"FILE: {path}")
    print(f"{'=' * 60}")

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum())