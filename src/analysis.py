# src/analysis.py
"""
Analysis Module
Dataset merging + master dataset persistence

Key fixes:
- Guarantees presence of 'ticker' and 'country' for sorting/merging
- Coerces dates safely (won't crash if strings/NaT)
- Handles macro datasets without 'country' by assuming company country
- Safe numeric aggregation on macro data
- Deduplicates company rows per (ticker, date) before merge
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetMerger:
    @staticmethod
    def merge_datasets(company_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n" + "=" * 80)
        logger.info("MERGING DATASETS")
        logger.info("=" * 80 + "\n")

        if company_df is None or company_df.empty:
            logger.warning("Company dataframe empty. Returning empty dataframe.")
            return pd.DataFrame()

        company_df = company_df.copy()
        macro_df = macro_df.copy() if macro_df is not None else pd.DataFrame()

        # Ensure keys exist
        if "ticker" not in company_df.columns:
            company_df["ticker"] = "MELI"
        if "country" not in company_df.columns or company_df["country"].isna().all():
            company_df["country"] = "BRA"

        company_df["date"] = pd.to_datetime(company_df.get("date"), errors="coerce")

        # If date missing, still allow a stable ordering
        if company_df["date"].isna().all():
            company_df["date"] = pd.date_range("2000-01-01", periods=len(company_df), freq="Q")

        company_df = (
            company_df.sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True)
        )

        # If no macro, just return company with basic defaults (caller can fill neutral regime)
        if macro_df is None or macro_df.empty:
            out = company_df.sort_values(["ticker", "date"]).reset_index(drop=True)
            logger.info("✓ No macro data provided; returning company dataset only")
            return out

        # Ensure macro has required columns
        macro_df["date"] = pd.to_datetime(macro_df.get("date"), errors="coerce")
        macro_df = macro_df[macro_df["date"].notna()].copy()

        if "country" not in macro_df.columns:
            # If macro lacks country tagging, assume it's all company country
            macro_df["country"] = company_df["country"].iloc[0]

        company_df["period"] = company_df["date"].dt.to_period("Q")
        macro_df["period"] = macro_df["date"].dt.to_period("Q")

        # Aggregate macro to quarterly per country
        numeric_cols = macro_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("period",)]  # period isn't numeric, but keep safe

        macro_agg = (
            macro_df.groupby(["period", "country"], as_index=False)[numeric_cols].mean()
            if numeric_cols
            else macro_df.groupby(["period", "country"], as_index=False).size().drop(columns=["size"], errors="ignore")
        )

        # Carry first observed categorical values (regime labels, etc.)
        categorical_cols = [
            c for c in macro_df.columns
            if c not in numeric_cols and c not in ["period", "country", "date"]
        ]
        if categorical_cols:
            cat_agg = macro_df.groupby(["period", "country"], as_index=False)[categorical_cols].first()
            macro_agg = pd.merge(macro_agg, cat_agg, on=["period", "country"], how="left")

        merged_df = pd.merge(company_df, macro_agg, on=["period", "country"], how="left", suffixes=("", "_macro"))
        merged_df = merged_df.drop(columns=["period"], errors="ignore")
        merged_df = merged_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info("✓ Merged dataset created")
        logger.info(f"  Rows: {len(merged_df)} | Cols: {len(merged_df.columns)}")
        return merged_df

    @staticmethod
    def save_master_dataset(df: pd.DataFrame, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"\n✓ Master dataset saved: {output_path}")
        logger.info(f"  Shape: {df.shape}")