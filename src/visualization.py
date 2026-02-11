# src/visualization.py
"""
Visualization Module - Institutional Grade

Key fixes:
- No legend warnings (only call legend when labeled artists exist)
- Consistent unit handling (decimals -> % in plots)
- WACC drawn from data (fallback 16%)
- Quarter labels fixed (no %q)
- Robust 'insufficient data' handling
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def _quarter_label(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "NA"
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def _wacc_series(df: pd.DataFrame) -> pd.Series:
    if "wacc" in df.columns and df["wacc"].notna().any():
        return pd.to_numeric(df["wacc"], errors="coerce").fillna(0.16)
    return pd.Series([0.16] * len(df), index=df.index)


def _safe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()


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
        create_enhanced_regime_heatmap(macro_df, output_dir / "regime_continuous.png")
        create_regime_probability_chart(macro_df, output_dir / "regime_probability.png")

    create_phase_space_diagram(df, output_dir / "phase_space_trajectory.png")
    create_capital_cycle_diagram(df, output_dir / "capital_cycle_traditional.png")

    create_cycle_decomposition_chart(df, output_dir / "cycle_decomposition.png")
    create_intensity_evolution(df, output_dir / "intensity_evolution.png")
    create_ep_persistence(df, output_dir / "ep_persistence.png")
    create_roic_evolution(df, output_dir / "roic_evolution.png")

    create_credit_scurve_chart(df, output_dir / "credit_scurve.png")
    create_credit_charts(df, output_dir)

    logger.info("✓ ALL VISUALIZATIONS COMPLETE\n")


def create_enhanced_regime_heatmap(macro_df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating enhanced regime heatmap...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    if "country" not in macro_df.columns:
        macro_df["country"] = "BRA"

    if "scarcity_index_normalized" in macro_df.columns and macro_df["scarcity_index_normalized"].notna().any():
        pivot = macro_df.pivot_table(
            index="country",
            columns=macro_df["date"].dt.to_period("Q"),
            values="scarcity_index_normalized",
            aggfunc="mean",
        )
        if not pivot.empty:
            sns.heatmap(
                pivot,
                cmap="RdYlGn_r",
                center=50,
                vmin=0,
                vmax=100,
                cbar_kws={"label": "Capital Scarcity Index (0=Abundant, 100=Severe)"},
                ax=axes[0],
                linewidths=0.3,
                annot=False,
            )
            axes[0].set_title("Continuous Capital Scarcity Index (Gradient)", fontsize=14, fontweight="bold")
        else:
            axes[0].text(0.5, 0.5, "No scarcity data", ha="center", va="center", transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, "No scarcity data", ha="center", va="center", transform=axes[0].transAxes)

    if "prob_scarcity" in macro_df.columns and macro_df["prob_scarcity"].notna().any():
        pivotp = macro_df.pivot_table(
            index="country",
            columns=macro_df["date"].dt.to_period("Q"),
            values="prob_scarcity",
            aggfunc="mean",
        )
        if not pivotp.empty:
            sns.heatmap(
                pivotp,
                cmap="Reds",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "P(Capital Scarcity Regime)"},
                ax=axes[1],
                linewidths=0.3,
                annot=False,
            )
            axes[1].set_title("Scarcity Regime Probability (Continuous)", fontsize=14, fontweight="bold")
        else:
            axes[1].text(0.5, 0.5, "No probability data", ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "No probability data", ha="center", va="center", transform=axes[1].transAxes)

    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_regime_probability_chart(macro_df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating regime probability chart...")
    fig, ax = plt.subplots(figsize=(14, 6))

    if "country" not in macro_df.columns:
        macro_df["country"] = "BRA"

    if "prob_scarcity" in macro_df.columns and macro_df["prob_scarcity"].notna().any():
        for country in macro_df["country"].dropna().unique():
            cdf = macro_df[macro_df["country"] == country].sort_values("date")
            ax.plot(cdf["date"], cdf["prob_scarcity"], marker="o", linewidth=2, markersize=4, label=str(country), alpha=0.8)

        ax.axhline(0.5, linestyle="--", linewidth=1.5, alpha=0.5, label="50% Threshold")
        ax.axhline(0.75, linestyle=":", linewidth=1.5, alpha=0.5, label="75% Severe")

    ax.set_ylim(0, 1)
    ax.set_title("Capital Scarcity Regime Probability Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("P(Capital Scarcity Regime)")
    _safe_legend(ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_phase_space_diagram(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating phase space trajectory diagram...")
    fig, ax = plt.subplots(figsize=(14, 10))

    needed = {"phase_x", "phase_y", "date"}
    if not needed.issubset(df.columns):
        ax.text(0.5, 0.5, "Phase space inputs missing", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    plot_df = df[df["phase_x"].notna() & df["phase_y"].notna()].sort_values("date").copy()
    if len(plot_df) < 2:
        ax.text(0.5, 0.5, "Insufficient data for phase space diagram", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # Color by scarcity if present; else by time
    if "scarcity_index_normalized" in plot_df.columns and plot_df["scarcity_index_normalized"].notna().any():
        colors = plot_df["scarcity_index_normalized"]
        cmap = "RdYlGn_r"
        label = "Capital Scarcity Index"
        vmin, vmax = 0, 100
    else:
        colors = np.arange(len(plot_df))
        cmap = "viridis"
        label = "Time Progression"
        vmin, vmax = None, None

    sc = ax.scatter(
        plot_df["phase_x"], plot_df["phase_y"],
        c=colors, s=200, cmap=cmap, alpha=0.85,
        edgecolors="black", linewidth=2, vmin=vmin, vmax=vmax
    )
    ax.plot(plot_df["phase_x"], plot_df["phase_y"], alpha=0.35, linewidth=2.5)

    ax.axhline(0, linestyle="--", linewidth=2, alpha=0.6)
    ax.axvline(0, linestyle="--", linewidth=2, alpha=0.6)

    first, last = plot_df.iloc[0], plot_df.iloc[-1]
    ax.scatter(first["phase_x"], first["phase_y"], s=400, marker="s", edgecolors="black", linewidth=3, label="Start")
    ax.scatter(last["phase_x"], last["phase_y"], s=500, marker="*", edgecolors="black", linewidth=3, label="Latest")
    ax.annotate(f"START\n{_quarter_label(first['date'])}", (first["phase_x"], first["phase_y"]), xytext=(12, 12), textcoords="offset points")
    ax.annotate(f"LATEST\n{_quarter_label(last['date'])}", (last["phase_x"], last["phase_y"]), xytext=(12, -18), textcoords="offset points")

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(label, rotation=270, labelpad=25)

    ax.set_xlabel("Investment Intensity (Z-Score)")
    ax.set_ylabel("ROIC (Z-Score)")
    ax.set_title("Capital Cycle Phase Space Trajectory (Institutional)")
    _safe_legend(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_capital_cycle_diagram(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating traditional capital cycle diagram...")
    fig, ax = plt.subplots(figsize=(12, 8))

    if "roic_ttm" not in df.columns or "total_capex_intensity" not in df.columns:
        ax.text(0.5, 0.5, "Inputs missing", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    plot_df = df[df["roic_ttm"].notna() & df["total_capex_intensity"].notna()].sort_values("date").copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    x = pd.to_numeric(plot_df["total_capex_intensity"], errors="coerce") * 100
    y = pd.to_numeric(plot_df["roic_ttm"], errors="coerce") * 100
    plot_df["time_order"] = range(len(plot_df))

    sc = ax.scatter(x, y, c=plot_df["time_order"], s=150, cmap="viridis", alpha=0.75, edgecolors="black", linewidth=1.5)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Time →", rotation=270, labelpad=20)

    median_capex = float(x.median())
    ax.axvline(median_capex, linestyle="--", linewidth=1.5, alpha=0.5, label=f"Median Capex ({median_capex:.1f}%)")

    wacc = float(_wacc_series(plot_df).iloc[-1]) * 100
    ax.axhline(wacc, linestyle="--", linewidth=1.5, alpha=0.6, label=f"WACC ({wacc:.1f}%)")

    first, last = plot_df.iloc[0], plot_df.iloc[-1]
    ax.scatter(first["total_capex_intensity"] * 100, first["roic_ttm"] * 100, s=220, marker="s", edgecolors="black", linewidth=2, label="Start")
    ax.scatter(last["total_capex_intensity"] * 100, last["roic_ttm"] * 100, s=260, marker="*", edgecolors="black", linewidth=2, label="Latest")

    ax.set_xlabel("Total Capex Intensity (% of Revenue)")
    ax.set_ylabel("ROIC (TTM, %)")
    ax.set_title("Capital Cycle Position (Traditional View)")
    _safe_legend(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_cycle_decomposition_chart(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating cycle decomposition chart...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if "roic_trend" in df.columns and df["roic_trend"].notna().any():
        ax = axes[0, 0]
        ax.plot(df["date"], df["roic_ttm"] * 100, label="ROIC TTM", alpha=0.5)
        ax.plot(df["date"], df["roic_trend"] * 100, label="Trend (HP)", linewidth=3)
        ax.set_title("ROIC: Structural Trend")
        ax.set_ylabel("ROIC (%)")
        _safe_legend(ax)

        ax = axes[0, 1]
        cyc = df.get("roic_cycle", pd.Series(index=df.index, data=np.nan)).fillna(0) * 100
        ax.bar(df["date"], cyc, alpha=0.7)
        ax.axhline(0, linewidth=2)
        ax.set_title("ROIC: Cyclical Deviation")
        ax.set_ylabel("pp")
    else:
        axes[0, 0].text(0.5, 0.5, "ROIC trend unavailable", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, "ROIC cycle unavailable", ha="center", va="center", transform=axes[0, 1].transAxes)

    if "total_capex_intensity_trend" in df.columns and df["total_capex_intensity_trend"].notna().any():
        ax = axes[1, 0]
        ax.plot(df["date"], df["total_capex_intensity"] * 100, label="Capex Intensity", alpha=0.5)
        ax.plot(df["date"], df["total_capex_intensity_trend"] * 100, label="Trend (HP)", linewidth=3)
        ax.set_title("Investment: Structural Trend")
        ax.set_ylabel("% of revenue")
        _safe_legend(ax)

        ax = axes[1, 1]
        cyc = df.get("total_capex_intensity_cycle", pd.Series(index=df.index, data=np.nan)).fillna(0) * 100
        ax.bar(df["date"], cyc, alpha=0.7)
        ax.axhline(0, linewidth=2)
        ax.set_title("Investment: Cyclical Deviation")
        ax.set_ylabel("pp")
    else:
        axes[1, 0].text(0.5, 0.5, "Capex trend unavailable", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, "Capex cycle unavailable", ha="center", va="center", transform=axes[1, 1].transAxes)

    for ax in axes.flat:
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_intensity_evolution(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating intensity evolution chart...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    if "growth_opex_intensity" in df.columns and df["growth_opex_intensity"].notna().any():
        axes[0].plot(df["date"], df["growth_opex_intensity"] * 100, marker="o", linewidth=2.5, label="Growth Opex (%)")
        if "rd_intensity" in df.columns and df["rd_intensity"].notna().any():
            axes[0].plot(df["date"], df["rd_intensity"] * 100, linestyle="--", label="R&D (%)", alpha=0.7)
        if "sm_intensity" in df.columns and df["sm_intensity"].notna().any():
            axes[0].plot(df["date"], df["sm_intensity"] * 100, linestyle="--", label="S&M (%)", alpha=0.7)
        axes[0].set_title("Growth Opex Investment Intensity")
        axes[0].set_ylabel("% of revenue")
        _safe_legend(axes[0])
    else:
        axes[0].text(0.5, 0.5, "Growth opex intensity unavailable", ha="center", va="center", transform=axes[0].transAxes)

    if "total_capex_intensity" in df.columns and df["total_capex_intensity"].notna().any():
        axes[1].plot(df["date"], df["total_capex_intensity"] * 100, marker="o", linewidth=2.5, label="Capex (%)")
        axes[1].set_title("Capex Intensity")
        axes[1].set_ylabel("% of revenue")
        _safe_legend(axes[1])
    else:
        axes[1].text(0.5, 0.5, "Capex intensity unavailable", ha="center", va="center", transform=axes[1].transAxes)

    if "total_growth_investment_intensity" in df.columns and df["total_growth_investment_intensity"].notna().any():
        axes[2].plot(df["date"], df["total_growth_investment_intensity"] * 100, marker="o", linewidth=2.5, label="Total Growth Investment (%)")
        axes[2].set_title("Total Growth Investment Intensity")
        axes[2].set_ylabel("% of revenue")
        _safe_legend(axes[2])
    else:
        axes[2].text(0.5, 0.5, "Total growth investment intensity unavailable", ha="center", va="center", transform=axes[2].transAxes)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_roic_evolution(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating ROIC evolution chart...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    wacc = _wacc_series(df) * 100

    if "roic_ttm" in df.columns and df["roic_ttm"].notna().any():
        ax1.plot(df["date"], df["roic_ttm"] * 100, linestyle="--", alpha=0.7, label="ROIC (Reported TTM, %)")
    if "adjusted_roic" in df.columns and df["adjusted_roic"].notna().any():
        ax1.plot(df["date"], df["adjusted_roic"] * 100, linewidth=2.8, label="ROIC (Adjusted, %)")

    if "roic_trend" in df.columns and df["roic_trend"].notna().any():
        ax1.plot(df["date"], df["roic_trend"] * 100, linestyle=":", alpha=0.8, label="Trend (HP, %)")

    ax1.plot(df["date"], wacc, linestyle="--", linewidth=2, alpha=0.7, label="WACC (%)")
    ax1.set_title("ROIC Evolution: Reported vs Adjusted")
    ax1.set_ylabel("ROIC (%)")
    _safe_legend(ax1)

    if "incremental_roic" in df.columns and df["incremental_roic"].notna().any():
        ax2.plot(df["date"], df["incremental_roic"] * 100, linestyle="--", alpha=0.7, label="Inc ROIC (Reported, %)")
    if "adjusted_incremental_roic" in df.columns and df["adjusted_incremental_roic"].notna().any():
        ax2.plot(df["date"], df["adjusted_incremental_roic"] * 100, linewidth=2.5, label="Inc ROIC (Adjusted, %)")

    ax2.plot(df["date"], wacc, linestyle="--", linewidth=2, alpha=0.7, label="WACC (%)")
    ax2.set_title("Incremental ROIC (3Y Rolling)")
    ax2.set_ylabel("Incremental ROIC (%)")
    _safe_legend(ax2)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_ep_persistence(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating EP persistence chart...")
    fig, ax = plt.subplots(figsize=(14, 6))

    if "economic_profit_ttm" in df.columns and df["economic_profit_ttm"].notna().any():
        ax.bar(df["date"], df["economic_profit_ttm"], alpha=0.7)
        ax.axhline(0, linewidth=2, alpha=0.7)

        if "ep_margin" in df.columns and df["ep_margin"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(df["date"], df["ep_margin"] * 100, marker="o", linewidth=2.2, label="EP Margin (%)")
            ax2.set_ylabel("EP Margin (%)")
            _safe_legend(ax2)
    else:
        ax.text(0.5, 0.5, "Economic profit series unavailable", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Economic Profit Persistence")
    ax.set_ylabel("Economic Profit (TTM, $)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_credit_scurve_chart(df: pd.DataFrame, output_path: Path) -> None:
    logger.info("Creating credit S-curve chart...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    if "credit_portfolio_log" in df.columns and df["credit_portfolio_log"].notna().any():
        axes[0].plot(df["date"], df["credit_portfolio_log"], marker="o", linewidth=2.3, label="log(credit)")
        axes[0].set_title("Credit Monetization (Log Scale)")
        _safe_legend(axes[0])
    else:
        axes[0].text(0.5, 0.5, "Credit log series unavailable", ha="center", va="center", transform=axes[0].transAxes)

    if "credit_portfolio_log_growth" in df.columns and df["credit_portfolio_log_growth"].notna().any():
        axes[1].plot(df["date"], df["credit_portfolio_log_growth"], marker="s", linewidth=2.0, label="Log growth")
        axes[1].axhline(0, alpha=0.5)
        axes[1].set_title("Credit Log Growth")
        _safe_legend(axes[1])
    else:
        axes[1].text(0.5, 0.5, "Credit log growth unavailable", ha="center", va="center", transform=axes[1].transAxes)

    if "credit_portfolio_log_accel" in df.columns and df["credit_portfolio_log_accel"].notna().any():
        axes[2].bar(df["date"], df["credit_portfolio_log_accel"].fillna(0), alpha=0.7)
        axes[2].axhline(0, linewidth=2)
        axes[2].set_title("Credit Growth Acceleration (Inflection)")
    else:
        axes[2].text(0.5, 0.5, "Credit acceleration unavailable", ha="center", va="center", transform=axes[2].transAxes)

    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_credit_charts(df: pd.DataFrame, output_dir: Path) -> None:
    logger.info("Creating credit portfolio charts...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Growth dynamics
    fig, ax = plt.subplots(figsize=(14, 6))
    if "credit_growth" in df.columns and df["credit_growth"].notna().any():
        ax.bar(df["date"], df["credit_growth"], alpha=0.6, label="YoY Growth (%)")
        ax.axhline(0, linewidth=1.5)

        if "credit_growth_accel" in df.columns and df["credit_growth_accel"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(df["date"], df["credit_growth_accel"], marker="o", linewidth=2, label="Acceleration (pp)")
            ax2.axhline(0, linestyle="--", alpha=0.5)
            _safe_legend(ax2)
        _safe_legend(ax)
    else:
        ax.text(0.5, 0.5, "Credit growth unavailable", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Credit Portfolio Growth Dynamics")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "credit_growth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Intensity in %
    fig, ax = plt.subplots(figsize=(14, 6))
    if "credit_intensity" in df.columns and df["credit_intensity"].notna().any():
        ax.plot(df["date"], df["credit_intensity"] * 100, marker="o", linewidth=2.5, label="Credit Intensity (%)")
        ax.axhline(20, linestyle="--", linewidth=1.5, alpha=0.6, label="20% Threshold")
        _safe_legend(ax)
    else:
        ax.text(0.5, 0.5, "Credit intensity unavailable", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Credit Deployment Intensity")
    ax.set_ylabel("% of revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "credit_intensity.png", dpi=300, bbox_inches="tight")
    plt.close()


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