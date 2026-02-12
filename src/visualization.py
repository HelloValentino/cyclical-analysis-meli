# src/visualization.py
"""
Visualization Module - Institutional Grade

Exports chart data as JSON for the interactive Plotly.js dashboard
and generates the CSV summary table.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _quarter_label(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "NA"
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def _wacc_series(df: pd.DataFrame) -> pd.Series:
    if "wacc" in df.columns and df["wacc"].notna().any():
        return pd.to_numeric(df["wacc"], errors="coerce").fillna(0.16)
    return pd.Series([0.16] * len(df), index=df.index)


def create_all_visualizations(df: pd.DataFrame, macro_df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING INSTITUTIONAL VISUALIZATIONS")
    logger.info("=" * 80 + "\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.sort_values("date")

    if macro_df is not None and not macro_df.empty:
        macro_df = macro_df.copy()
        macro_df["date"] = pd.to_datetime(macro_df.get("date"), errors="coerce")
        macro_df = macro_df[macro_df["date"].notna()].sort_values("date")

    # Export JSON for interactive dashboard
    data_dir = output_dir.parent / "data"
    export_chart_data_json(df, macro_df, data_dir)

    logger.info("✓ ALL VISUALIZATIONS COMPLETE\n")


# ─────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────

def create_summary_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    logger.info("Creating summary table...")
    if df is None or df.empty:
        return pd.DataFrame()

    latest = df.iloc[-1]

    def fmt_pct(x):
        return "N/A" if pd.isna(x) else f"{x * 100:.1f}"

    wacc = float(latest.get("wacc", 0.16))

    rows = [
        ("ROIC (Reported TTM, %)", fmt_pct(latest.get("roic_ttm"))),
        ("ROIC (Adjusted, %)", fmt_pct(latest.get("adjusted_roic"))),
        ("WACC (%)", f"{wacc * 100:.1f}"),
        ("Spread (Adj - WACC, pp)", "N/A" if pd.isna(latest.get("adjusted_roic")) else f"{(latest.get('adjusted_roic') - wacc) * 100:.1f}"),
        ("Total Capex Intensity (%)", fmt_pct(latest.get("total_capex_intensity"))),
        ("Total Growth Investment Intensity (%)", fmt_pct(latest.get("total_growth_investment_intensity"))),
        ("Scarcity Index (0-100)", "N/A" if pd.isna(latest.get("scarcity_index_normalized")) else f"{latest.get('scarcity_index_normalized'):.0f}"),
        ("P(Scarcity)", "N/A" if pd.isna(latest.get("prob_scarcity")) else f"{latest.get('prob_scarcity'):.2f}"),
        ("Phase Quadrant", str(latest.get("phase_quadrant", "N/A"))),
    ]

    out = pd.DataFrame(rows, columns=["Metric", "Latest"])
    out.to_csv(output_path, index=False)
    logger.info(f"  ✓ Saved: {Path(output_path).name}")
    return out


# ─────────────────────────────────────────────────────────────
# JSON Data Export for Interactive Dashboard
# ─────────────────────────────────────────────────────────────

def _ser(series, mult=1):
    """Convert a pandas Series to a JSON-safe list, multiplying by mult."""
    return [None if (pd.isna(v) or not np.isfinite(v)) else round(float(v) * mult, 6) for v in series]


def _dates(series):
    return [d.isoformat() if pd.notna(d) else None for d in pd.to_datetime(series, errors="coerce")]


def export_chart_data_json(df: pd.DataFrame, macro_df, output_dir: Path) -> None:
    """Export all chart data as JSON files for the interactive Plotly.js dashboard."""
    logger.info("Exporting chart data to JSON...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = _dates(df["date"])
    wacc = _ser(_wacc_series(df), 100)

    # 1. Regime heatmap (from macro_df)
    regime = {"scarcity": {}, "probability": {}}
    if macro_df is not None and not macro_df.empty:
        mdf = macro_df.copy()
        if "country" not in mdf.columns:
            mdf["country"] = "BRA"
        mdf["quarter"] = mdf["date"].dt.to_period("Q").astype(str)
        for val_col, key in [("scarcity_index_normalized", "scarcity"), ("prob_scarcity", "probability")]:
            if val_col in mdf.columns and mdf[val_col].notna().any():
                piv = mdf.pivot_table(index="country", columns="quarter", values=val_col, aggfunc="mean")
                regime[key] = {"countries": piv.index.tolist(), "quarters": piv.columns.tolist(),
                               "values": [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in piv.values]}
    _write(output_dir / "regime_heatmap.json", regime)

    # 2. Regime probability time-series (from macro_df)
    regime_ts = {"series": []}
    if macro_df is not None and not macro_df.empty:
        mdf = macro_df.copy()
        if "country" not in mdf.columns:
            mdf["country"] = "BRA"
        if "prob_scarcity" in mdf.columns:
            from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
            for country in mdf["country"].dropna().unique():
                cdf = mdf[mdf["country"] == country].sort_values("date")
                # Aggregate to quarterly for a readable chart
                cdf = cdf.set_index("date")
                qdf = cdf[["prob_scarcity"]].resample("QE").mean().dropna().reset_index()
                prob = qdf["prob_scarcity"].values
                lowess_smooth = [None] * len(prob)
                if len(prob) >= 4:
                    xs = np.arange(len(prob))
                    result = _lowess(prob, xs, frac=0.2, return_sorted=False)
                    lowess_smooth = [round(float(v), 6) for v in result]
                regime_ts["series"].append({"country": str(country), "dates": _dates(qdf["date"]),
                                            "prob_scarcity": _ser(qdf["prob_scarcity"]),
                                            "lowess": lowess_smooth})
    _write(output_dir / "regime_probability.json", regime_ts)

    # 3. Phase space trajectory
    ps_df = df[df["phase_x"].notna() & df["phase_y"].notna()].sort_values("date") if {"phase_x","phase_y"}.issubset(df.columns) else df.head(0)
    phase = {"dates": _dates(ps_df["date"]), "phase_x": _ser(ps_df["phase_x"]),
             "phase_y": _ser(ps_df["phase_y"]),
             "scarcity": _ser(ps_df["scarcity_index_normalized"]) if "scarcity_index_normalized" in ps_df.columns else [],
             "labels": [_quarter_label(d) for d in ps_df["date"]]}
    _write(output_dir / "phase_space.json", phase)

    # 4. Capital cycle traditional
    cc_df = df[df["roic_ttm"].notna() & df["total_capex_intensity"].notna()].sort_values("date") if {"roic_ttm","total_capex_intensity"}.issubset(df.columns) else df.head(0)
    cap = {"dates": _dates(cc_df["date"]), "capex_intensity": _ser(cc_df["total_capex_intensity"], 100),
           "roic_ttm": _ser(cc_df["roic_ttm"], 100),
           "wacc": round(float(_wacc_series(cc_df).iloc[-1] * 100), 2) if len(cc_df) else 16.0,
           "labels": [_quarter_label(d) for d in cc_df["date"]]}
    _write(output_dir / "capital_cycle.json", cap)

    # 5. Cycle decomposition (4 panels)
    cycle = {"dates": dates,
             "roic_ttm": _ser(df.get("roic_ttm", pd.Series(dtype=float)), 100),
             "roic_trend": _ser(df.get("roic_trend", pd.Series(dtype=float)), 100),
             "roic_cycle": _ser(df.get("roic_cycle", pd.Series(dtype=float)).fillna(0), 100),
             "capex_intensity": _ser(df.get("total_capex_intensity", pd.Series(dtype=float)), 100),
             "capex_trend": _ser(df.get("total_capex_intensity_trend", pd.Series(dtype=float)), 100),
             "capex_cycle": _ser(df.get("total_capex_intensity_cycle", pd.Series(dtype=float)).fillna(0), 100)}
    _write(output_dir / "cycle_decomposition.json", cycle)

    # 6. Intensity evolution (3 panels)
    intensity = {"dates": dates,
                 "growth_opex": _ser(df.get("growth_opex_intensity", pd.Series(dtype=float)), 100),
                 "rd": _ser(df.get("rd_intensity", pd.Series(dtype=float)), 100),
                 "sm": _ser(df.get("sm_intensity", pd.Series(dtype=float)), 100),
                 "capex": _ser(df.get("total_capex_intensity", pd.Series(dtype=float)), 100),
                 "total_growth": _ser(df.get("total_growth_investment_intensity", pd.Series(dtype=float)), 100)}
    _write(output_dir / "intensity_evolution.json", intensity)

    # 7. ROIC evolution (2 panels)
    roic = {"dates": dates, "wacc": wacc,
            "roic_ttm": _ser(df.get("roic_ttm", pd.Series(dtype=float)), 100),
            "adjusted_roic": _ser(df.get("adjusted_roic", pd.Series(dtype=float)), 100),
            "roic_trend": _ser(df.get("roic_trend", pd.Series(dtype=float)), 100),
            "incremental_roic": _ser(df.get("incremental_roic", pd.Series(dtype=float)), 100),
            "adjusted_incremental_roic": _ser(df.get("adjusted_incremental_roic", pd.Series(dtype=float)), 100)}
    _write(output_dir / "roic_evolution.json", roic)

    # 8. EP persistence
    ep = {"dates": dates,
          "economic_profit_ttm": _ser(df.get("economic_profit_ttm", pd.Series(dtype=float))),
          "ep_margin": _ser(df.get("ep_margin", pd.Series(dtype=float)), 100)}
    _write(output_dir / "ep_persistence.json", ep)

    # 9. Credit S-curve (3 panels)
    scurve = {"dates": dates,
              "credit_log": _ser(df.get("credit_portfolio_log", pd.Series(dtype=float))),
              "credit_log_growth": _ser(df.get("credit_portfolio_log_growth", pd.Series(dtype=float))),
              "credit_log_accel": _ser(df.get("credit_portfolio_log_accel", pd.Series(dtype=float)))}
    _write(output_dir / "credit_scurve.json", scurve)

    # 10. Credit growth
    cg = {"dates": dates,
          "credit_growth": _ser(df.get("credit_growth", pd.Series(dtype=float))),
          "credit_growth_accel": _ser(df.get("credit_growth_accel", pd.Series(dtype=float)))}
    _write(output_dir / "credit_growth.json", cg)

    # 11. Credit intensity
    ci = {"dates": dates,
          "credit_intensity": _ser(df.get("credit_intensity", pd.Series(dtype=float)), 100)}
    _write(output_dir / "credit_intensity.json", ci)

    logger.info(f"  ✓ Exported 11 JSON files to {output_dir}")


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
