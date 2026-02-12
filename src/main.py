# src/main.py
from __future__ import annotations

import os
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_acquisition import MacroDataFetcher, CompanyDataLoader
from calculations import VPCalculator
from analysis import DatasetMerger
from visualization import create_all_visualizations, create_summary_table
from advanced_analysis import apply_all_advanced_analytics, InstitutionalMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ensure_dirs(config: dict) -> None:
    for path_key in ["outputs_figures", "outputs_tables", "data_processed"]:
        Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("\n" + "=" * 80)
    print("VP CAPITAL CYCLE ANALYSIS - MERCADOLIBRE")
    print("INSTITUTIONAL-GRADE FRAMEWORK WITH VP/HOLT ADJUSTED ROIC")
    print("=" * 80 + "\n")

    root = Path(__file__).resolve().parent.parent
    config_path = root / "config.yaml"
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.chdir(root)
    _ensure_dirs(config)

    # STEP 1: MACRO
    macro_cache = Path(config["paths"]["data_raw"]) / "macro" / "macro_master.csv"
    if macro_cache.exists():
        logger.info("STEP 1: LOADING CACHED MACRO DATA (%s)", macro_cache)
        macro_df = pd.read_csv(str(macro_cache), parse_dates=["date"])
        logger.info("  ✓ Loaded %d rows from cache", len(macro_df))
    else:
        logger.info("STEP 1: FETCHING MACRO DATA (no cache found)")
        macro_fetcher = MacroDataFetcher(
            start_date=config["time_periods"]["start_date"],
            end_date=config["time_periods"]["end_date"],
            output_dir=str(Path(config["paths"]["data_raw"]) / "macro"),
        )
        try:
            macro_df = macro_fetcher.fetch_all()
        except Exception as e:
            logger.error("Macro fetch failed: %s", e, exc_info=True)
            macro_df = None

    if macro_df is None or macro_df.empty:
        logger.warning("No macro data available - continuing with company data only")
        macro_df = None

    # STEP 2: COMPANY
    logger.info("STEP 2: LOADING COMPANY FINANCIALS")
    company_loader = CompanyDataLoader(data_dir=str(Path(config["paths"]["data_raw"]) / "company"))
    csv_path = Path(config["paths"]["data_raw"]) / "company" / "meli_financials.csv"

    if not csv_path.exists():
        logger.error("Financial data not found at: %s", csv_path)
        logger.info("Run: python sec_data_extractor.py")
        return

    try:
        meli_df = company_loader.load_financials(str(csv_path), default_ticker="MELI")
    except Exception as e:
        logger.error("Error loading financials: %s", e, exc_info=True)
        logger.error("No financial data loaded. Exiting.")
        return

    if meli_df.empty:
        logger.error("No financial data loaded. Exiting.")
        return

    # STEP 3: CALCULATIONS
    logger.info("STEP 3: CALCULATING VP METRICS (WITH ADJUSTED ROIC)")
    calculator = VPCalculator(
        wacc=config["parameters"]["base_wacc"],
        country_risk=config["parameters"]["country_risk_premium"],
        rd_amortization_years=config["parameters"]["rd_amortization_years"],
    )
    meli_df = calculator.calculate_all_metrics(meli_df)

    # STEP 4: ADVANCED ANALYTICS
    logger.info("STEP 4: APPLYING INSTITUTIONAL ANALYTICS")
    if macro_df is None:
        # advanced analytics should still run (company-only)
        meli_df, _ = apply_all_advanced_analytics(meli_df, None)
    else:
        meli_df, macro_df = apply_all_advanced_analytics(meli_df, macro_df)

    # STEP 5: MERGE
    logger.info("STEP 5: MERGING DATASETS")
    if macro_df is not None and not macro_df.empty:
        master_df = DatasetMerger.merge_datasets(meli_df, macro_df)

        # Neutral defaults if macro-derived cols missing
        if "scarcity_index_normalized" not in master_df.columns:
            master_df["scarcity_index_normalized"] = 50.0
        master_df["scarcity_index_normalized"] = master_df["scarcity_index_normalized"].fillna(50.0)

        if "prob_scarcity" not in master_df.columns:
            master_df["prob_scarcity"] = 0.5
        master_df["prob_scarcity"] = master_df["prob_scarcity"].fillna(0.5)

        if "regime_probabilistic" not in master_df.columns:
            master_df["regime_probabilistic"] = "normal"
        master_df["regime_probabilistic"] = master_df["regime_probabilistic"].fillna("normal")

        logger.info("✓ Filled missing macro data with neutral defaults")
    else:
        master_df = meli_df.copy()
        master_df["scarcity_index_normalized"] = 50.0
        master_df["prob_scarcity"] = 0.5
        master_df["regime_probabilistic"] = "normal"
        logger.warning("No macro data - using neutral regime defaults")

    # SAVE MASTER
    master_path = Path(config["paths"]["data_processed"]) / "master_dataset.parquet"
    DatasetMerger.save_master_dataset(master_df, str(master_path))

    # STEP 6: SUMMARY
    logger.info("STEP 6: GENERATING INSTITUTIONAL SUMMARY")
    institutional_summary = InstitutionalMetrics.generate_institutional_summary(master_df)

    # STEP 7: VISUALS
    logger.info("STEP 7: GENERATING VISUALIZATIONS")
    fig_dir = Path(config["paths"]["outputs_figures"])
    create_all_visualizations(master_df, (macro_df if macro_df is not None else None), fig_dir)

    # STEP 8: SUMMARY TABLE
    logger.info("STEP 8: GENERATING SUMMARY TABLE")
    table_path = Path(config["paths"]["outputs_tables"]) / "summary_metrics.csv"
    summary_df = create_summary_table(master_df, table_path)
    print(summary_df.to_string(index=False))

    # PRINT INSIGHTS
    latest = master_df.iloc[-1]
    wacc = float(latest.get("wacc", config["parameters"]["base_wacc"] + 0.04))
    adj_roic = float(latest.get("adjusted_roic", np.nan))

    print("\n" + "=" * 80)
    print("INSTITUTIONAL INSIGHTS")
    print("=" * 80)
    print(f"Capital Cycle Phase: {institutional_summary.get('phase_quadrant', 'N/A')}")
    print(f"Scarcity Regime: {institutional_summary.get('regime_classification', 'N/A')}")
    print(f"Scarcity Probability: {institutional_summary.get('prob_scarcity_regime', 0):.1%}")
    print(f"Super-Cycle Signal: {'CONFIRMED ✓' if institutional_summary.get('super_cycle_signal') else 'Not Detected'}")

    if np.isfinite(adj_roic):
        spread_pp = (adj_roic - wacc) * 100
        print(f"\nAdjusted ROIC: {adj_roic*100:.1f}%  |  WACC: {wacc*100:.1f}%  |  Spread: {spread_pp:.1f}pp")

    print("\n✓ ANALYSIS COMPLETE\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        logger.error("Analysis failed: %s", str(e), exc_info=True)
        raise