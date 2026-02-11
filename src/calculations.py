# src/calculations.py
"""
Calculations Module
All VP-aligned metric calculations with VP/HOLT ADJUSTED ROIC

Key fixes / hardening:
- Enforces presence of 'ticker' + 'date' (safe defaults)
- Coerces numeric columns robustly
- Avoids groupby boolean-index crash patterns
- Robust cash tax rate when taxes_paid sparse:
    prefer taxes_paid -> fallback tax_expense -> default 25%
- Flow-consistent VP/HOLT adjusted ROIC
- Deduplicates by (ticker, date) and sorts
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_columns(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df


def _coerce_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class VPCalculator:
    """
    VP Capital Cycle Calculator
    Includes VP/HOLT Adjusted ROIC for platform companies
    """

    def __init__(
        self,
        wacc: float = 0.12,
        country_risk: Optional[Dict[str, float]] = None,
        rd_amortization_years: float = 4.0,
    ):
        self.base_wacc = float(wacc)
        self.country_risk = country_risk or {"ARG": 0.08, "BRA": 0.04, "MEX": 0.02, "LATAM": 0.05}
        self.rd_amortization_years = float(rd_amortization_years)

    def calculate_all_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATING ALL VP METRICS (WITH ADJUSTED ROIC)")
        logger.info("=" * 80 + "\n")

        if df is None or df.empty:
            logger.warning("Input dataframe is empty. Returning empty dataframe.")
            return pd.DataFrame()

        df = df.copy()
        df = _ensure_columns(
            df,
            {
                "ticker": "MELI",
                "country": "BRA",
            },
        )
        df = _coerce_datetime(df, "date")

        # If date is missing entirely, create a synthetic index-based time ordering (won't crash downstream)
        if "date" not in df.columns or df["date"].isna().all():
            df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="Q")

        # De-duplicate and sort to prevent weird rolling artifacts
        df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)

        numeric_candidates = [
            "revenue",
            "cogs",
            "sales_marketing",
            "rd_expense",
            "ga_expense",
            "depreciation",
            "ebit",
            "taxes_paid",
            "tax_expense",
            "cash",
            "current_assets",
            "current_liabilities",
            "short_term_debt",
            "long_term_debt",
            "net_ppe",
            "gross_ppe",
            "accumulated_depreciation",
            "ppe_additions",
            "intangible_additions",
            "credit_portfolio",
            "total_assets",
            "software_investment",
        ]
        df = _coerce_numeric(df, numeric_candidates)

        df = self._calculate_invested_capital(df)
        df = self._calculate_returns(df)
        df = self._calculate_intensity(df)
        df = self._calculate_adjusted_roic(df)

        logger.info("\n✓ All VP metrics calculated successfully\n")
        return df

    def _calculate_invested_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating Invested Capital (Simplified)...")
        df = df.copy()

        df = _ensure_columns(
            df,
            {
                "current_assets": 0.0,
                "cash": 0.0,
                "current_liabilities": 0.0,
                "short_term_debt": 0.0,
                "net_ppe": 0.0,
                "accumulated_depreciation": 0.0,
                "rd_expense": 0.0,
            },
        )
        df = _coerce_numeric(df, ["current_assets", "cash", "current_liabilities", "short_term_debt", "net_ppe", "accumulated_depreciation", "rd_expense"])
        df[["current_assets", "cash", "current_liabilities", "short_term_debt", "net_ppe", "accumulated_depreciation", "rd_expense"]] = (
            df[["current_assets", "cash", "current_liabilities", "short_term_debt", "net_ppe", "accumulated_depreciation", "rd_expense"]]
            .fillna(0.0)
            .astype(float)
        )

        df["nwc"] = (df["current_assets"] - df["cash"]) - (df["current_liabilities"] - df["short_term_debt"])

        if "gross_ppe" not in df.columns or df["gross_ppe"].isna().all():
            df["gross_ppe"] = df["net_ppe"] + df["accumulated_depreciation"]
        df["gross_ppe"] = pd.to_numeric(df["gross_ppe"], errors="coerce").fillna(0.0)

        # R&D capitalization (rolling sum / life in years; consistent with quarterly flows)
        amort_q = max(1, int(round(self.rd_amortization_years * 4)))
        df["capitalized_rd"] = (
            df.groupby("ticker")["rd_expense"]
            .transform(lambda x: x.rolling(amort_q, min_periods=1).sum())
            / self.rd_amortization_years
        )

        df["invested_capital"] = df["nwc"] + df["gross_ppe"] + df["capitalized_rd"]

        df["invested_capital_avg"] = (
            df.groupby("ticker")["invested_capital"].transform(lambda x: (x + x.shift(1)) / 2.0)
        ).fillna(df["invested_capital"])

        logger.info("✓ Invested Capital calculated")
        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating Return Metrics (Simplified)...")
        df = df.copy()

        df = _ensure_columns(
            df,
            {
                "ebit": 0.0,
                "taxes_paid": np.nan,
                "tax_expense": np.nan,
                "rd_expense": 0.0,
                "capitalized_rd": 0.0,
                "invested_capital_avg": 0.0,
                "invested_capital": 0.0,
                "revenue": 0.0,
                "country": "BRA",
            },
        )
        df = _coerce_numeric(df, ["ebit", "taxes_paid", "tax_expense", "rd_expense", "capitalized_rd", "invested_capital_avg", "invested_capital", "revenue"])

        DEFAULT_TAX = 0.25
        ebit_pos = df["ebit"].astype(float).clip(lower=0.01)

        taxes_paid = df["taxes_paid"]
        tax_expense = df["tax_expense"]

        cash_rate = (taxes_paid / ebit_pos).where(taxes_paid.notna())
        accrual_rate = (tax_expense / ebit_pos).where(tax_expense.notna())

        df["cash_tax_rate"] = cash_rate.fillna(accrual_rate).fillna(DEFAULT_TAX).clip(0.0, 0.40)

        df["rd_amortization"] = pd.to_numeric(df["capitalized_rd"], errors="coerce").fillna(0.0) / self.rd_amortization_years

        # NOPAT with R&D capitalization adjustment (VP/HOLT style)
        df["nopat"] = df["ebit"].fillna(0.0) * (1 - df["cash_tax_rate"]) + df["rd_expense"].fillna(0.0) - df["rd_amortization"].fillna(0.0)

        df["nopat_ttm"] = df.groupby("ticker")["nopat"].transform(lambda x: x.rolling(4, min_periods=1).sum())
        df["revenue_ttm"] = df.groupby("ticker")["revenue"].transform(lambda x: x.rolling(4, min_periods=1).sum())

        ic_avg = df["invested_capital_avg"].replace(0, np.nan)
        df["roic"] = df["nopat"] / ic_avg
        df["roic_ttm"] = df["nopat_ttm"] / ic_avg

        df["roic_delta"] = df.groupby("ticker")["roic_ttm"].diff(4)
        df["roic_delta_3y"] = df.groupby("ticker")["roic_ttm"].diff(12)

        df["nopat_change"] = df.groupby("ticker")["nopat_ttm"].diff(12)
        df["ic_change"] = df.groupby("ticker")["invested_capital"].diff(12)
        df["incremental_roic"] = df["nopat_change"] / df["ic_change"].replace(0, np.nan)

        df["wacc"] = df["country"].map(lambda c: self.base_wacc + float(self.country_risk.get(c, 0.0))).astype(float)

        df["economic_profit"] = df["invested_capital"] * (df["roic"] - df["wacc"])
        df["economic_profit_ttm"] = df["invested_capital"] * (df["roic_ttm"] - df["wacc"])
        df["ep_margin"] = df["economic_profit_ttm"] / df["revenue_ttm"].replace(0, np.nan)

        df["ep_growth"] = df.groupby("ticker")["economic_profit_ttm"].pct_change(4, fill_method=None)
        df["roic_spread_bps"] = (df["roic_ttm"] - df["wacc"]) * 10000

        logger.info("✓ Return metrics calculated")
        return df

    def _calculate_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating Capital Intensity Metrics...")
        df = df.copy()

        df = _ensure_columns(
            df,
            {
                "depreciation": 0.0,
                "ppe_additions": 0.0,
                "intangible_additions": 0.0,
                "revenue": 0.0,
                "sales_marketing": 0.0,
                "rd_expense": 0.0,
                "nwc": 0.0,
                "nopat_ttm": 0.0,
            },
        )
        df = _coerce_numeric(df, ["depreciation", "ppe_additions", "intangible_additions", "revenue", "sales_marketing", "rd_expense", "nwc", "nopat_ttm"])
        df[["depreciation", "ppe_additions", "intangible_additions", "revenue", "sales_marketing", "rd_expense", "nwc", "nopat_ttm"]] = (
            df[["depreciation", "ppe_additions", "intangible_additions", "revenue", "sales_marketing", "rd_expense", "nwc", "nopat_ttm"]]
            .fillna(0.0)
            .astype(float)
        )

        if "revenue_ttm" not in df.columns:
            df["revenue_ttm"] = df.groupby("ticker")["revenue"].transform(lambda x: x.rolling(4, min_periods=1).sum())
        rev_ttm = df["revenue_ttm"].replace(0, np.nan)

        df["maintenance_capex"] = df["depreciation"]
        df["growth_capex"] = (df["ppe_additions"] + df["intangible_additions"] - df["maintenance_capex"]).clip(lower=0)

        df["growth_capex_ttm"] = df.groupby("ticker")["growth_capex"].transform(lambda x: x.rolling(4, min_periods=1).sum())
        df["maintenance_capex_ttm"] = df.groupby("ticker")["maintenance_capex"].transform(lambda x: x.rolling(4, min_periods=1).sum())

        df["growth_capex_intensity"] = df["growth_capex_ttm"] / rev_ttm
        df["maintenance_capex_intensity"] = df["maintenance_capex_ttm"] / rev_ttm
        df["total_capex_intensity"] = df["growth_capex_intensity"] + df["maintenance_capex_intensity"]

        denom = (df["growth_capex_ttm"] + df["maintenance_capex_ttm"]).replace(0, np.nan)
        df["growth_capex_share"] = df["growth_capex_ttm"] / denom

        df["growth_opex"] = df["sales_marketing"] + df["rd_expense"]
        df["growth_opex_ttm"] = df.groupby("ticker")["growth_opex"].transform(lambda x: x.rolling(4, min_periods=1).sum())
        df["rd_ttm"] = df.groupby("ticker")["rd_expense"].transform(lambda x: x.rolling(4, min_periods=1).sum())
        df["sm_ttm"] = df.groupby("ticker")["sales_marketing"].transform(lambda x: x.rolling(4, min_periods=1).sum())

        df["growth_opex_intensity"] = df["growth_opex_ttm"] / rev_ttm
        df["rd_intensity"] = df["rd_ttm"] / rev_ttm
        df["sm_intensity"] = df["sm_ttm"] / rev_ttm

        df["total_growth_investment"] = df["growth_capex_ttm"] + df["growth_opex_ttm"]
        df["total_growth_investment_intensity"] = df["total_growth_investment"] / rev_ttm
        df["growth_investment_intensity_delta"] = df.groupby("ticker")["total_growth_investment_intensity"].diff(4)

        if "credit_portfolio" in df.columns:
            df["credit_portfolio"] = pd.to_numeric(df["credit_portfolio"], errors="coerce")
            df["credit_growth"] = df.groupby("ticker")["credit_portfolio"].pct_change(4, fill_method=None) * 100
            df["credit_growth_accel"] = df.groupby("ticker")["credit_growth"].diff(4)
            df["credit_intensity"] = df["credit_portfolio"] / rev_ttm
            df["credit_growth_qoq"] = df.groupby("ticker")["credit_portfolio"].pct_change(fill_method=None) * 100
            df["credit_growth_absolute"] = df.groupby("ticker")["credit_portfolio"].diff(4)
            if "invested_capital" in df.columns:
                df["credit_ic_ratio"] = df["credit_portfolio"] / pd.to_numeric(df["invested_capital"], errors="coerce").replace(0, np.nan)

        df["nwc_change"] = df.groupby("ticker")["nwc"].diff()
        df["nwc_investment"] = df["nwc_change"].clip(lower=0)
        df["nwc_investment_ttm"] = df.groupby("ticker")["nwc_investment"].transform(lambda x: x.rolling(4, min_periods=1).sum())

        df["total_reinvestment"] = df["total_growth_investment"] + df["nwc_investment_ttm"]
        df["reinvestment_rate"] = df["total_reinvestment"] / df["nopat_ttm"].replace(0, np.nan)
        df["reinvestment_rate_capped"] = df["reinvestment_rate"].clip(upper=5.0)

        logger.info("✓ Capital intensity metrics calculated")
        return df

    def _calculate_adjusted_roic(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating VP/HOLT Adjusted ROIC (flow-consistent)...")
        df = df.copy()

        df = _ensure_columns(
            df,
            {
                "sales_marketing": 0.0,
                "intangible_additions": 0.0,
                "software_investment": np.nan,
                "credit_portfolio": 0.0,
                "cash": 0.0,
                "ebit": 0.0,
                "cash_tax_rate": np.nan,
                "wacc": self.base_wacc + self.country_risk.get("BRA", 0.0),
                "nwc": 0.0,
                "capitalized_rd": 0.0,
                "gross_ppe": np.nan,
                "net_ppe": 0.0,
                "accumulated_depreciation": 0.0,
                "cogs": 0.0,
                "rd_expense": 0.0,
                "ga_expense": 0.0,
                "revenue": 0.0,
            },
        )

        num_cols = [c for c in df.columns if c not in {"ticker", "country", "date"}]
        df = _coerce_numeric(df, num_cols)

        # Software investment proxy if missing
        if df["software_investment"].isna().all() or (df["software_investment"].fillna(0).abs().sum() == 0):
            df["software_investment"] = df["intangible_additions"]
        df["software_investment"] = pd.to_numeric(df["software_investment"], errors="coerce").fillna(0.0)

        # Build gross PPE if missing
        if "gross_ppe" not in df.columns or df["gross_ppe"].isna().all():
            df["gross_ppe"] = df["net_ppe"].fillna(0.0) + df["accumulated_depreciation"].fillna(0.0)
        df["gross_ppe"] = pd.to_numeric(df["gross_ppe"], errors="coerce").fillna(0.0)

        SM_LIFE_Q = 8
        SW_LIFE_Q = 16
        DEFAULT_TAX = 0.25

        def _asset_and_amort(spend_series: pd.Series, life_q: int) -> pd.DataFrame:
            spend = pd.to_numeric(spend_series, errors="coerce").fillna(0.0).astype(float).values
            asset = np.zeros(len(spend), dtype=float)
            amort = np.zeros(len(spend), dtype=float)
            for t in range(len(spend)):
                prev_asset = asset[t - 1] if t > 0 else 0.0
                amort[t] = prev_asset / float(life_q)
                asset[t] = prev_asset - amort[t] + spend[t]
            return pd.DataFrame({"asset": asset, "amort": amort}, index=spend_series.index)

        sm_out = df.groupby("ticker", group_keys=False)["sales_marketing"].apply(lambda s: _asset_and_amort(s, SM_LIFE_Q))
        df["capitalized_sm"] = sm_out["asset"].values
        df["sm_amortization"] = sm_out["amort"].values

        sw_out = df.groupby("ticker", group_keys=False)["software_investment"].apply(lambda s: _asset_and_amort(s, SW_LIFE_Q))
        df["capitalized_software"] = sw_out["asset"].values
        df["software_amortization"] = sw_out["amort"].values

        if "operating_expenses" not in df.columns:
            df["operating_expenses"] = (
                df["cogs"].fillna(0.0)
                + df["sales_marketing"].fillna(0.0)
                + df["rd_expense"].fillna(0.0)
                + df["ga_expense"].fillna(0.0)
            )

        # Operating cash need proxy; ensures excess cash doesn't explode in edge cases
        operating_cash_need = df["operating_expenses"].fillna(0.0)
        excess_cash = (df["cash"].fillna(0.0) - operating_cash_need).clip(lower=0.0)

        credit_offset = np.minimum(excess_cash, df["credit_portfolio"].fillna(0.0) * 0.15)
        df["operating_credit_assets"] = (df["credit_portfolio"].fillna(0.0) - credit_offset).clip(lower=0.0)

        df["cash_tax_rate"] = df["cash_tax_rate"].fillna(DEFAULT_TAX).clip(0.0, 0.40)

        # Adjust EBIT: capitalize S&M and software investment like economic assets
        df["adjusted_ebit"] = (
            df["ebit"].fillna(0.0)
            + df["sales_marketing"].fillna(0.0)
            - df["sm_amortization"].fillna(0.0)
            + df["software_investment"].fillna(0.0)
            - df["software_amortization"].fillna(0.0)
        )

        df["adjusted_nopat"] = df["adjusted_ebit"] * (1.0 - df["cash_tax_rate"])
        df["adjusted_nopat_ttm"] = df.groupby("ticker")["adjusted_nopat"].transform(lambda x: x.rolling(4, min_periods=1).sum())

        df["adjusted_invested_capital"] = (
            df["gross_ppe"].fillna(0.0)
            + df["capitalized_rd"].fillna(0.0)
            + df["capitalized_sm"].fillna(0.0)
            + df["capitalized_software"].fillna(0.0)
            + df["nwc"].fillna(0.0)
            + df["operating_credit_assets"].fillna(0.0)
        )

        df["adjusted_invested_capital_avg"] = (
            df.groupby("ticker")["adjusted_invested_capital"].transform(lambda x: (x + x.shift(1)) / 2.0)
        ).fillna(df["adjusted_invested_capital"])

        denom = df["adjusted_invested_capital_avg"].replace(0, np.nan)
        df["adjusted_roic"] = df["adjusted_nopat_ttm"] / denom

        df["adjusted_nopat_ttm_change"] = df.groupby("ticker")["adjusted_nopat_ttm"].diff(12)
        df["adjusted_ic_change"] = df.groupby("ticker")["adjusted_invested_capital"].diff(12)
        df["adjusted_incremental_roic"] = df["adjusted_nopat_ttm_change"] / df["adjusted_ic_change"].replace(0, np.nan)

        df["adjusted_economic_profit"] = df["adjusted_invested_capital"] * (df["adjusted_roic"] - df["wacc"])
        return df


if __name__ == "__main__":
    print("calculations.py loaded successfully (no standalone execution).")