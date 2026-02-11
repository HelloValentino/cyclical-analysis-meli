# src/advanced_analysis.py
"""
Advanced Analysis Module
Institutional-grade capital cycle decomposition and regime modeling

Key fixes:
- Scarcity index safe defaults when macro missing or components degenerate
- Robust z-score functions (avoid NaN/inf propagation)
- Cycle decomposition guarded for short histories
- Phase space robust if intensities missing/constant
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousRegimeAnalyzer:
    def __init__(self):
        self.scarcity_weights = {
            "inflation_vol": 0.30,
            "fx_vol": 0.25,
            "real_rate_repression": 0.25,
            "credit_collapse": 0.20,
        }

    def calculate_continuous_scarcity_index(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating continuous capital scarcity index...")
        df = df.copy()

        if df.empty:
            df["scarcity_index_continuous"] = 0.0
            df["scarcity_index_normalized"] = 50.0
            logger.warning("  ⚠ Empty macro df -> scarcity neutral 50")
            return df

        if "country" not in df.columns:
            df["country"] = "BRA"

        components = {}

        # Inflation vol
        if "cpi" in df.columns and df["cpi"].notna().any():
            df["inflation_vol_raw"] = df.groupby("country")["cpi"].transform(lambda x: x.rolling(12, min_periods=6).std())
            components["inflation_vol"] = self._z_score(df["inflation_vol_raw"])

        # FX vol (first matching column)
        fx_vol_cols = [c for c in df.columns if "_vol_12m" in c and "usd_" in c]
        if fx_vol_cols and df[fx_vol_cols[0]].notna().any():
            df["fx_vol_raw"] = df[fx_vol_cols[0]]
            components["fx_vol"] = self._z_score(df["fx_vol_raw"])

        # Real-rate repression
        if "real_rate" in df.columns and df["real_rate"].notna().any():
            df["real_rate_repression_raw"] = -df["real_rate"]
            components["real_rate_repression"] = self._z_score(df["real_rate_repression_raw"])

        # Credit collapse proxy (only if cpi exists; safe placeholder)
        if "cpi" in df.columns and df["cpi"].notna().any():
            df["credit_growth_proxy"] = df.groupby("country")["cpi"].transform(lambda x: -x.rolling(4, min_periods=2).mean())
            components["credit_collapse"] = self._z_score(df["credit_growth_proxy"])

        if components:
            df["scarcity_index_continuous"] = 0.0
            for k, v in components.items():
                df["scarcity_index_continuous"] += v * float(self.scarcity_weights.get(k, 0.25))

            # Normalize robustly to 0..100
            q05 = df["scarcity_index_continuous"].quantile(0.05)
            q95 = df["scarcity_index_continuous"].quantile(0.95)
            span = q95 - q05

            if not np.isfinite(span) or abs(span) < 1e-9:
                df["scarcity_index_normalized"] = 50.0
            else:
                df["scarcity_index_normalized"] = ((df["scarcity_index_continuous"] - q05) / span * 100).clip(0, 100)

            logger.info(
                f"  ✓ Scarcity index range: "
                f"{df['scarcity_index_normalized'].min():.1f} - {df['scarcity_index_normalized'].max():.1f}"
            )
        else:
            df["scarcity_index_continuous"] = 0.0
            df["scarcity_index_normalized"] = 50.0
            logger.warning("  ⚠ Insufficient macro components -> scarcity neutral 50")

        return df

    def estimate_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Estimating regime probabilities...")
        df = df.copy()

        if "scarcity_index_continuous" not in df.columns:
            df = self.calculate_continuous_scarcity_index(df)

        # Logistic mapping
        k = 2.0
        threshold = 0.0
        df["prob_scarcity"] = expit(k * (df["scarcity_index_continuous"] - threshold)).astype(float)

        df["regime_probabilistic"] = pd.cut(
            df["prob_scarcity"],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=["abundant", "normal", "scarce", "severe"],
            include_lowest=True,
        ).astype(str)

        return df

    @staticmethod
    def _z_score(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        mean = s.mean()
        std = s.std()
        if std is None or std == 0 or not np.isfinite(std):
            return s * 0.0
        z = (s - mean) / std
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


class CycleDecomposer:
    def __init__(self, lambda_hp: float = 1600):
        self.lambda_hp = lambda_hp

    def decompose_cycle(self, df: pd.DataFrame, var: str, trend_name: str = None, cycle_name: str = None) -> pd.DataFrame:
        df = df.copy()

        if var not in df.columns or df[var].isna().all():
            return df

        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
        except ImportError:
            logger.error("statsmodels not installed. Run: pip install statsmodels")
            return df

        valid = df[var].dropna()
        if len(valid) < 8:
            return df

        cycle, trend = hpfilter(valid, lamb=self.lambda_hp)

        tcol = trend_name or f"{var}_trend"
        ccol = cycle_name or f"{var}_cycle"

        df.loc[valid.index, tcol] = trend
        df.loc[valid.index, ccol] = cycle
        return df

    def decompose_all_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Decomposing capital cycle metrics into trend + cycle...")
        df = df.copy()

        df = self.decompose_cycle(df, "roic_ttm", trend_name="roic_trend", cycle_name="roic_cycle")
        df = self.decompose_cycle(df, "total_capex_intensity")
        df = self.decompose_cycle(df, "growth_opex_intensity")
        df = self.decompose_cycle(df, "economic_profit_ttm")
        df = self.decompose_cycle(df, "reinvestment_rate")
        df = self.decompose_cycle(df, "incremental_roic")

        logger.info("✓ Cycle decomposition complete\n")
        return df


class LogTransformer:
    @staticmethod
    def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying log transforms for S-curve detection...")
        df = df.copy()

        log_vars = ["credit_portfolio", "economic_profit_ttm", "revenue", "invested_capital", "total_assets"]
        for var in log_vars:
            if var in df.columns and df[var].notna().any():
                base = pd.to_numeric(df[var], errors="coerce").fillna(0.0).clip(lower=0.0)
                df[f"{var}_log"] = np.log1p(base)
                df[f"{var}_log_growth"] = df.groupby("ticker")[f"{var}_log"].diff() * 100
                df[f"{var}_log_accel"] = df.groupby("ticker")[f"{var}_log_growth"].diff()
            else:
                df[f"{var}_log"] = np.nan
                df[f"{var}_log_growth"] = np.nan
                df[f"{var}_log_accel"] = np.nan
        return df


class PhaseSpaceAnalyzer:
    @staticmethod
    def create_phase_space_coords(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating phase space coordinates...")
        df = df.copy()

        if "ticker" not in df.columns:
            df["ticker"] = "MELI"

        roic = pd.to_numeric(df.get("roic_ttm"), errors="coerce")
        if roic is not None and roic.notna().any():
            df["roic_ttm_z"] = PhaseSpaceAnalyzer._z(roic)
        else:
            df["roic_ttm_z"] = 0.0

        tgi = pd.to_numeric(df.get("total_growth_investment_intensity"), errors="coerce")
        if tgi is not None and tgi.notna().any():
            df["tgi_z"] = PhaseSpaceAnalyzer._z(tgi)
            df["phase_x"] = df["tgi_z"]
        else:
            capex = pd.to_numeric(df.get("total_capex_intensity"), errors="coerce")
            df["total_capex_intensity_z"] = PhaseSpaceAnalyzer._z(capex.fillna(0.0) if capex is not None else pd.Series([0.0] * len(df)))
            df["phase_x"] = df["total_capex_intensity_z"]

        df["phase_y"] = df["roic_ttm_z"]
        df["phase_quadrant"] = df.apply(lambda r: PhaseSpaceAnalyzer._get_quadrant(r["phase_x"], r["phase_y"]), axis=1)
        return df

    @staticmethod
    def _z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        std = s.std()
        if std is None or std == 0 or not np.isfinite(std):
            return s * 0.0
        z = (s - s.mean()) / std
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    @staticmethod
    def _get_quadrant(x: float, y: float) -> str:
        if pd.isna(x) or pd.isna(y):
            return "unknown"
        if y >= 0 and x >= 0:
            return "super_cycle"
        if y >= 0 and x < 0:
            return "value_creation"
        if y < 0 and x < 0:
            return "capital_abundance"
        return "capital_destruction"


class InstitutionalMetrics:
    @staticmethod
    def generate_institutional_summary(df: pd.DataFrame) -> Dict:
        logger.info("Generating institutional summary metrics...")
        if df is None or df.empty:
            return {}

        latest = df.iloc[-1]
        prev_4q = df.iloc[-5] if len(df) >= 5 else latest

        summary = {
            "scarcity_index": latest.get("scarcity_index_normalized", np.nan),
            "prob_scarcity_regime": latest.get("prob_scarcity", np.nan),
            "regime_classification": str(latest.get("regime_probabilistic", "unknown")),
            "roic_z_score": latest.get("roic_ttm_z", np.nan),
            "investment_z_score": latest.get("phase_x", np.nan),
            "phase_quadrant": latest.get("phase_quadrant", "unknown"),
            "roic_trend": latest.get("roic_trend", np.nan),
            "roic_cyclical": latest.get("roic_cycle", np.nan),
            "investment_trend": latest.get("total_capex_intensity_trend", np.nan),
            "investment_cyclical": latest.get("total_capex_intensity_cycle", np.nan),
            "credit_log_growth": latest.get("credit_portfolio_log_growth", np.nan),
            "credit_acceleration": latest.get("credit_portfolio_log_accel", np.nan),
            "credit_inflection_status": "post-inflection"
            if latest.get("credit_portfolio_log_accel", 0) < 0
            else "pre-inflection",
            "ep_margin": latest.get("ep_margin", np.nan),
            "ep_trend": latest.get("economic_profit_ttm_trend", np.nan),
            "ep_4q_change": latest.get("economic_profit_ttm", 0) - prev_4q.get("economic_profit_ttm", 0),
            "reinvestment_rate": latest.get("reinvestment_rate", np.nan),
            "growth_investment_intensity": latest.get("total_growth_investment_intensity", np.nan),
            "incremental_roic": latest.get("incremental_roic", np.nan),
        }

        # Thresholds in decimals (0.20 = 20%)
        summary["super_cycle_signal"] = (
            summary["phase_quadrant"] == "super_cycle"
            and np.isfinite(summary["roic_trend"])
            and summary["roic_trend"] > 0.20
            and summary.get("prob_scarcity_regime", 0) > 0.5
        )

        return summary


def apply_all_advanced_analytics(company_df: pd.DataFrame, macro_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING INSTITUTIONAL-GRADE ANALYTICS")
    logger.info("=" * 80 + "\n")

    # Macro regimes (safe if macro incomplete)
    regime_analyzer = ContinuousRegimeAnalyzer()
    if macro_df is not None and not macro_df.empty:
        macro_df = regime_analyzer.calculate_continuous_scarcity_index(macro_df)
        macro_df = regime_analyzer.estimate_regime_probabilities(macro_df)

    # Company cycle work
    decomposer = CycleDecomposer(lambda_hp=1600)
    company_df = decomposer.decompose_all_metrics(company_df)

    company_df = LogTransformer.apply_log_transforms(company_df)
    company_df = PhaseSpaceAnalyzer.create_phase_space_coords(company_df)

    return company_df, macro_df