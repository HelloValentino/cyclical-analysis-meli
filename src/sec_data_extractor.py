
# sec_data_extractor.py
from __future__ import annotations

import os
import json
import time
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class ExtractConfig:
    ticker: str = "MELI"
    cik: str = "0001099590"  # MELI
    cache_path: Path = Path("data/raw/sec_cache/meli_companyfacts.json")
    output_csv: Path = Path("data/raw/company/meli_financials.csv")

    # Keep last N quarters (most recent)
    max_quarters: Optional[int] = 40

    # Q4 inference (FY - Q1-Q3) when Q4 missing
    infer_q4: bool = True

    timeout: int = 30
    max_retries: int = 3
    sleep_between: float = 0.6


# ----------------------------
# SEC identity / headers
# ----------------------------
def sec_headers() -> Dict[str, str]:
    email = os.getenv("SEC_EDGAR_EMAIL", "").strip()
    name = os.getenv("SEC_EDGAR_COMPANY_NAME", "").strip()
    if not (email and name):
        logger.warning(
            "SEC EDGAR identity not configured. Set SEC_EDGAR_EMAIL and SEC_EDGAR_COMPANY_NAME in .env for SEC compliance."
        )
        ua = "VPResearch/1.0 (no-email-provided)"
    else:
        ua = f"{name} ({email})"
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


# ----------------------------
# Fetch / cache
# ----------------------------
def fetch_companyfacts(cfg: ExtractConfig) -> dict:
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.cache_path.exists():
        try:
            obj = json.loads(cfg.cache_path.read_text(encoding="utf-8"))
            logger.info("✓ Loaded SEC data from cache")
            return obj
        except Exception:
            logger.warning("Cache corrupted. Refetching...")

    url = SEC_COMPANYFACTS.format(cik=cfg.cik.zfill(10))
    headers = sec_headers()

    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=cfg.timeout)
            r.raise_for_status()
            obj = r.json()
            cfg.cache_path.write_text(json.dumps(obj), encoding="utf-8")
            logger.info("✓ Successfully fetched SEC data")
            return obj
        except Exception as e:
            last_err = e
            logger.warning("SEC fetch attempt %d/%d failed: %s", attempt, cfg.max_retries, e)
            time.sleep(cfg.sleep_between * attempt)

    raise RuntimeError(f"Failed to fetch SEC Company Facts after {cfg.max_retries} retries: {last_err}")


# ----------------------------
# Metric mapping
# ----------------------------
METRICS: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueServicesNet",
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],
    "sales_marketing": ["SellingAndMarketingExpense"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "ga_expense": ["GeneralAndAdministrativeExpense"],
    "depreciation": ["DepreciationAndAmortization"],
    "ebit": ["OperatingIncomeLoss"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "current_tax_expense": ["CurrentIncomeTaxExpenseBenefit"],
    "taxes_paid": ["IncomeTaxesPaid"],
    "net_income": ["NetIncomeLoss"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "ppe_additions": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "intangible_additions": ["PaymentsToAcquireIntangibleAssets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "current_assets": ["AssetsCurrent"],
    "net_ppe": ["PropertyPlantAndEquipmentNet"],
    "accumulated_depreciation": [
        "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAccumulatedDepreciationAndAmortization",
    ],
    "gross_ppe": [
        "PropertyPlantAndEquipmentGross",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization",
    ],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill"],
    "total_assets": ["Assets"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "short_term_debt": [
        "DebtCurrent",
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermLoansPayable",
    ],
    "equity": ["StockholdersEquity"],
    "credit_portfolio": [
        "NotesReceivableNet",
        "LoansAndLeasesReceivableNetReportedAmount",
        "NotesAndLoansReceivableNetCurrent",
        "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
    ],
}

# Flow metrics: should not be negative (for sanity checks)
NON_NEGATIVE_METRICS = {
    "revenue", "cogs", "sales_marketing", "rd_expense", "ga_expense",
    "depreciation", "tax_expense", "current_tax_expense", "taxes_paid",
    "operating_cf", "ppe_additions", "intangible_additions",
}


# ----------------------------
# Helpers: extract facts
# ----------------------------
def _pick_unit(units_dict: dict) -> Optional[str]:
    if not isinstance(units_dict, dict) or not units_dict:
        return None
    if "USD" in units_dict:
        return "USD"
    return next(iter(units_dict.keys()))


def _to_float(x) -> float:
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def extract_metric_facts(companyfacts: dict, tag: str) -> pd.DataFrame:
    """
    Normalized fact table:
      columns: tag, end, fy, fp, form, filed, val
    Dedups best record per (end, fp): 10-Q preferred over 10-K over others; latest filed preferred.
    """
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    node = facts.get(tag)
    if not node:
        return pd.DataFrame()

    unit_key = _pick_unit(node.get("units", {}))
    if not unit_key:
        return pd.DataFrame()

    rows = []
    for item in node["units"].get(unit_key, []):
        end = item.get("end")
        fp = item.get("fp")
        fy = item.get("fy")
        form = (item.get("form") or "").strip()
        filed = item.get("filed")
        val = _to_float(item.get("val"))
        start = item.get("start")

        if not end or not fp or not fy:
            continue

        rows.append(
            {
                "tag": tag,
                "start": start,
                "end": end,
                "fy": int(fy) if str(fy).isdigit() else fy,
                "fp": str(fp).upper().strip(),
                "form": form,
                "filed": filed,
                "val": val,
                "unit": unit_key,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "val"]).copy()

    df["is_10q"] = (df["form"] == "10-Q").astype(int)
    df["is_10k"] = (df["form"] == "10-K").astype(int)
    df["form_rank"] = df["is_10q"] * 3 + df["is_10k"] * 2

    df = df.sort_values(["end", "fp", "form_rank", "filed"], ascending=[True, True, False, False])
    df = df.drop_duplicates(subset=["end", "fp"], keep="first")

    return df[["tag", "start", "end", "fy", "fp", "form", "filed", "val", "unit"]].reset_index(drop=True)


def choose_best_tag(companyfacts: dict, tags: List[str]) -> Tuple[pd.DataFrame, str]:
    best_df = pd.DataFrame()
    best_tag = ""
    for tag in tags:
        df = extract_metric_facts(companyfacts, tag)
        if df.empty:
            continue
        if best_df.empty or len(df) > len(best_df):
            best_df = df
            best_tag = tag
    return best_df, best_tag


def _detect_outlier_indices(series: pd.Series, factor: float = 50.0) -> set:
    """Return indices where values appear to be filed in the wrong unit.

    Detects *dip-and-recover* and *spike-and-revert* patterns in log₁₀ space.
    A sudden magnitude change of >``factor``× followed by a reversal of similar
    size indicates the intervening values are outliers (e.g. filed in thousands
    instead of units).

    This is robust to **consecutive** bad values because it tracks magnitude
    *transitions* rather than comparing each value to a rolling statistic that
    can itself become corrupted when the majority of neighbours are bad.
    """
    if len(series) < 3:
        return set()
    vals = series.dropna().sort_index()
    if len(vals) < 3:
        return set()

    # Work in log₁₀ space so multiplicative discontinuities become additive
    log_vals = np.log10(vals.abs().replace(0, np.nan)).dropna()
    if len(log_vals) < 3:
        return set()

    threshold = np.log10(factor)
    indices = list(log_vals.index)
    log_arr = log_vals.values
    diffs = np.diff(log_arr)  # diffs[i] = log_arr[i+1] − log_arr[i]

    # Collect big jumps: (first_index_after_jump, direction)
    jumps: list = []
    for i, d in enumerate(diffs):
        if d < -threshold:
            jumps.append((i + 1, "drop"))
        elif d > threshold:
            jumps.append((i + 1, "rise"))

    # Pair consecutive *opposite* jumps → the values between them are outliers
    bad: set = set()
    j = 0
    while j < len(jumps) - 1:
        pos1, t1 = jumps[j]
        pos2, t2 = jumps[j + 1]
        if t1 != t2:  # opposite directions = dip-and-recover or spike-and-revert
            for k in range(pos1, pos2):
                bad.add(indices[k])
            j += 2  # consume both jumps
        else:
            j += 1  # same direction – genuine level shift, skip
    return bad


def merge_all_tags(companyfacts: dict, tags: List[str]) -> Tuple[pd.DataFrame, str]:
    """Merge facts from multiple XBRL tags.  Earlier tags have higher priority.

    Includes an **outlier guard**: after the initial priority-based dedup, any
    values that form a dip-and-recover or spike-and-revert pattern in log-space
    (magnitude change >50×) are replaced by the next-priority tag's value for
    the same (end, fp) slot.  This catches SEC XBRL filing errors where values
    are reported in the wrong unit, even when multiple consecutive values are
    corrupted.
    """
    all_dfs = []
    used_tags = []
    for tag in tags:
        df = extract_metric_facts(companyfacts, tag)
        if not df.empty:
            all_dfs.append(df)
            used_tags.append(tag)
    if not all_dfs:
        return pd.DataFrame(), ""
    if len(all_dfs) == 1:
        return all_dfs[0], used_tags[0]

    combined = pd.concat(all_dfs, ignore_index=True)
    tag_priority = {t: i for i, t in enumerate(used_tags)}
    combined["_pri"] = combined["tag"].map(tag_priority)
    combined = combined.sort_values(["end", "fp", "_pri", "filed"], ascending=[True, True, True, False])

    # --- First pass: pick highest-priority value per (end, fp) ---
    deduped = combined.drop_duplicates(subset=["end", "fp"], keep="first").copy()

    # --- Outlier guard: detect discontinuities in the chosen series ---
    deduped = deduped.sort_values("end").reset_index(drop=True)
    bad_idx = _detect_outlier_indices(deduped.set_index(deduped.index)["val"])

    if bad_idx:
        # Build a lookup of all alternative values per (end, fp)
        alt_lookup: Dict[Tuple, list] = {}
        for _, row in combined.iterrows():
            key = (row["end"], row["fp"])
            alt_lookup.setdefault(key, []).append(row)

        replaced = 0
        for idx in bad_idx:
            row = deduped.loc[idx]
            key = (row["end"], row["fp"])
            alts = alt_lookup.get(key, [])
            # Try next-priority alternatives
            for alt in sorted(alts, key=lambda r: r["_pri"]):
                if alt["tag"] == row["tag"]:
                    continue
                alt_val = alt["val"]
                if pd.notna(alt_val) and np.isfinite(alt_val):
                    deduped.loc[idx, "val"] = alt_val
                    deduped.loc[idx, "tag"] = alt["tag"]
                    replaced += 1
                    break
        if replaced:
            logger.info("  ⚠ Outlier guard: replaced %d suspect values with next-priority tag data", replaced)

    deduped = deduped.drop(columns=["_pri"], errors="ignore")
    return deduped.reset_index(drop=True), "+".join(used_tags)


# ----------------------------
# Q4 inference (robust + no over-inference)
# ----------------------------
def _looks_ytd(q1: float, q2: float, q3: float) -> bool:
    """
    Heuristic: if Q2 and Q3 behave like cumulative/YTD values, they will usually be
    increasing and substantially larger than prior quarters.
    """
    if not all(np.isfinite([q1, q2, q3])):
        return False
    if q1 <= 0:
        return False
    # Monotone increasing + meaningfully bigger suggests YTD
    return (q2 > q1 * 1.15) and (q3 > q2 * 1.08)


def _to_quarterly_from_ytd(q1: float, q2: float, q3: float) -> Tuple[float, float, float]:
    # Convert cumulative into quarterly flows
    return q1, (q2 - q1), (q3 - q2)


def infer_q4_for_metric(panel_q: pd.DataFrame, facts_df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, int]:
    """
    Infer missing Q4 from FY - (Q1+Q2+Q3), using facts_df (which includes FY rows).
    Also reclassifies short-span "FY" entries (≤100 days, ending in December) as Q4 directly.
    """
    if facts_df.empty:
        return panel_q, 0

    facts_df = facts_df.copy()
    facts_df["end"] = pd.to_datetime(facts_df["end"], errors="coerce")

    # Reclassify short-span FY entries as Q4 — ONLY if they end in December.
    # Short-span FY entries ending in Mar/Jun/Sep are comparative-period data, NOT Q4.
    if "start" in facts_df.columns:
        facts_df["start"] = pd.to_datetime(facts_df["start"], errors="coerce")
        fy_mask = facts_df["fp"] == "FY"
        has_start = facts_df["start"].notna()
        span = (facts_df["end"] - facts_df["start"]).dt.days
        short = span.between(1, 100)  # ≤ ~3 months
        end_month_dec = facts_df["end"].dt.month == 12
        reclassify = fy_mask & has_start & short & end_month_dec
        if reclassify.any():
            facts_df.loc[reclassify, "fp"] = "Q4"
            logger.info("  ✓ Reclassified %d short-span FY entries as Q4", int(reclassify.sum()))

    # Also populate panel directly from any newly-reclassified Q4 rows
    q4_direct = facts_df[facts_df["fp"] == "Q4"].dropna(subset=["end", "val"])
    if not q4_direct.empty and metric in panel_q.columns:
        for _, r in q4_direct.iterrows():
            end = pd.to_datetime(r["end"])
            val = float(r["val"]) if np.isfinite(r["val"]) else np.nan
            if not np.isfinite(val):
                continue
            mask = (panel_q["date"] == end) & (panel_q["period_type"] == "Q4")
            if mask.any() and panel_q.loc[mask, metric].isna().all():
                panel_q.loc[mask, metric] = val

    out = panel_q.copy()
    applied = 0

    # Calendar year from end date (EDGAR's `fy` field is the filing year, NOT the data year)
    facts_df["_cal_yr"] = facts_df["end"].dt.year

    # FY rows: only genuine full-year entries (span > 300 days or no start available)
    fy_rows = facts_df[facts_df["fp"] == "FY"].dropna(subset=["end", "val"])
    if "start" in fy_rows.columns:
        fy_span = (fy_rows["end"] - fy_rows["start"]).dt.days
        fy_rows = fy_rows[fy_span.isna() | (fy_span > 300)]
    if fy_rows.empty:
        return out, 0

    # Quarter facts from facts_df — filter to correct end-month per quarter.
    # EDGAR comparative data can have e.g. fp=Q1 with end=Dec (full-year value), which
    # would poison inference.  Q1→Mar, Q2→Jun, Q3→Sep, Q4→Dec.
    _fp_month = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}
    q_rows = facts_df[facts_df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])].dropna(subset=["end", "val"])
    q_rows = q_rows[q_rows.apply(lambda r: r["end"].month == _fp_month.get(r["fp"], 0), axis=1)]
    if q_rows.empty:
        return out, 0

    # Map calendar year -> FY end date & FY value (take latest FY by end)
    fy_map = (
        fy_rows.sort_values("end")
        .drop_duplicates(subset=["_cal_yr"], keep="last")
        .set_index("_cal_yr")[["end", "val"]]
        .rename(columns={"end": "fy_end", "val": "fy_val"})
    )

    # Build quarter values per calendar year
    q_map = (
        q_rows.sort_values("end")
        .drop_duplicates(subset=["_cal_yr", "fp"], keep="last")
        .pivot_table(index="_cal_yr", columns="fp", values="val", aggfunc="last")
    )

    for cal_yr, row in fy_map.iterrows():
        fy_end = pd.to_datetime(row["fy_end"])
        fy_val = float(row["fy_val"]) if np.isfinite(row["fy_val"]) else np.nan
        if not np.isfinite(fy_val):
            continue

        if cal_yr not in q_map.index:
            continue

        q1 = q_map.loc[cal_yr].get("Q1", np.nan)
        q2 = q_map.loc[cal_yr].get("Q2", np.nan)
        q3 = q_map.loc[cal_yr].get("Q3", np.nan)
        q4_existing = q_map.loc[cal_yr].get("Q4", np.nan)

        # If Q4 exists in facts, do not infer
        if np.isfinite(q4_existing):
            continue

        # Need Q1-Q3 present
        if not (np.isfinite(q1) and np.isfinite(q2) and np.isfinite(q3)):
            continue

        # Detect YTD and convert if needed
        if _looks_ytd(float(q1), float(q2), float(q3)):
            q1_q, q2_q, q3_q = _to_quarterly_from_ytd(float(q1), float(q2), float(q3))
        else:
            q1_q, q2_q, q3_q = float(q1), float(q2), float(q3)

        qsum = q1_q + q2_q + q3_q
        q4 = fy_val - qsum

        if not np.isfinite(q4):
            continue

        # Sanity checks
        if metric in NON_NEGATIVE_METRICS and q4 < 0:
            continue

        # Avoid extreme implied Q4 relative to recent quarters (robust against mismatched units)
        base_vals = [abs(v) for v in [q1_q, q2_q, q3_q] if np.isfinite(v)]
        med = float(np.median(base_vals)) if base_vals else np.nan
        if np.isfinite(med) and med > 0 and abs(q4) > 10.0 * med:
            continue

        # Only fill if the panel has the Q4 row and it's missing for this metric
        mask = (out["date"] == fy_end) & (out["period_type"] == "Q4")
        if mask.any():
            if out.loc[mask, metric].isna().all():
                out.loc[mask, metric] = q4
                applied += 1
        else:
            # If this quarter end isn't in the panel, skip (do not invent dates)
            continue

    return out, applied


# ----------------------------
# Panel construction
# ----------------------------
def build_quarter_index(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    # Use QE to avoid pandas deprecation warning
    q_ends = pd.date_range(min_date, max_date, freq="QE")
    out = pd.DataFrame({"date": q_ends})
    out["period_type"] = out["date"].dt.quarter.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    return out


def snap_balance_sheet(panel: pd.DataFrame, col: str) -> None:
    panel[col] = pd.to_numeric(panel[col], errors="coerce").ffill()


def main() -> None:
    cfg = ExtractConfig()

    print("\n" + "=" * 80)
    logger.info("HIGH-QUALITY MELI DATA EXTRACTION")
    logger.info("=" * 80 + "\n")

    companyfacts = fetch_companyfacts(cfg)

    logger.info("Extracting all financial metrics with comprehensive field coverage...")
    logger.info("=" * 80)

    # For each metric: choose best tag, keep its full facts table (includes FY and Q1-Q4)
    metric_facts: Dict[str, pd.DataFrame] = {}
    chosen_tags: Dict[str, str] = {}

    total_points = 0
    for metric, tags in METRICS.items():
        df, tag = merge_all_tags(companyfacts, tags)
        if df.empty:
            logger.warning("  ✗ %-28s : NOT FOUND (tried %d fields)", metric, len(tags))
            continue
        metric_facts[metric] = df
        chosen_tags[metric] = tag
        logger.info("  ✓ %-28s : %4d data points from '%s'", metric, int(len(df)), tag)
        total_points += int(len(df))

    logger.info("=" * 80)
    logger.info("Total data points extracted: %d", total_points)

    if not metric_facts:
        raise RuntimeError("No metrics extracted. Check CIK/tags/SEC connectivity.")

    # Determine quarter-end bounds from Q1-Q4 ends across all metrics
    ends = []
    for df in metric_facts.values():
        qdf = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]
        ends.extend(qdf["end"].dropna().tolist())

    if not ends:
        raise RuntimeError("No quarterly endpoints extracted (Q1-Q4).")

    min_end = pd.to_datetime(min(ends))
    max_end = pd.to_datetime(max(ends))

    panel = build_quarter_index(min_end, max_end)

    # Populate quarterly values
    for metric, df in metric_facts.items():
        panel[metric] = np.nan

        qdf = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
        if qdf.empty:
            continue

        # best record already deduped; set into panel by (end, fp)
        for _, r in qdf.iterrows():
            end = pd.to_datetime(r["end"])
            fp = str(r["fp"]).upper().strip()
            val = float(r["val"]) if np.isfinite(r["val"]) else np.nan
            if not np.isfinite(val):
                continue
            panel.loc[(panel["date"] == end) & (panel["period_type"] == fp), metric] = val

    # --- Convert YTD cumulative cash-flow items to standalone quarterly values ---
    # SEC EDGAR reports CF items (taxes_paid, operating_cf, ppe_additions, etc.) as YTD.
    # Detect by checking if the raw facts have start dates beginning Jan 1 for Q2/Q3.
    CF_METRICS_YTD = {
        "taxes_paid", "operating_cf", "ppe_additions", "intangible_additions",
    }
    for metric in CF_METRICS_YTD:
        if metric not in metric_facts or metric not in panel.columns:
            continue
        facts_df = metric_facts[metric]
        # Check if Q2/Q3 entries have start=Jan-01 (YTD pattern)
        q23 = facts_df[facts_df["fp"].isin(["Q2", "Q3"])].copy()
        if q23.empty or "start" not in q23.columns:
            continue
        q23["start"] = pd.to_datetime(q23["start"], errors="coerce")
        jan_starts = q23["start"].dropna().apply(lambda d: d.month == 1 and d.day <= 2)
        if jan_starts.sum() < max(1, len(jan_starts) * 0.5):
            continue  # Not YTD pattern

        logger.info("  ✓ Detected YTD pattern for %s — converting to standalone quarterly", metric)

        # Group by calendar year and convert YTD → standalone
        for year in panel["date"].dt.year.unique():
            yr_mask = panel["date"].dt.year == year
            q1_mask = yr_mask & (panel["period_type"] == "Q1")
            q2_mask = yr_mask & (panel["period_type"] == "Q2")
            q3_mask = yr_mask & (panel["period_type"] == "Q3")

            q1_val = panel.loc[q1_mask, metric].values
            q2_val = panel.loc[q2_mask, metric].values
            q3_val = panel.loc[q3_mask, metric].values

            q1_v = float(q1_val[0]) if len(q1_val) and np.isfinite(q1_val[0]) else np.nan
            q2_v = float(q2_val[0]) if len(q2_val) and np.isfinite(q2_val[0]) else np.nan
            q3_v = float(q3_val[0]) if len(q3_val) and np.isfinite(q3_val[0]) else np.nan

            # Q2_standalone = Q2_ytd - Q1_ytd; Q3_standalone = Q3_ytd - Q2_ytd
            if np.isfinite(q2_v) and np.isfinite(q1_v):
                panel.loc[q2_mask, metric] = q2_v - q1_v
            if np.isfinite(q3_v) and np.isfinite(q2_v):
                panel.loc[q3_mask, metric] = q3_v - q2_v

    # --- Populate balance sheet items from FY rows (FY end = Q4 end for Dec fiscal year) ---
    BS_METRICS = {
        "cash", "current_assets", "net_ppe", "accumulated_depreciation", "gross_ppe",
        "intangibles", "total_assets", "current_liabilities", "long_term_debt",
        "equity", "credit_portfolio",
    }
    for metric in BS_METRICS:
        if metric not in panel.columns or metric not in metric_facts:
            continue
        facts_df = metric_facts[metric]
        fy_rows = facts_df[facts_df["fp"] == "FY"].dropna(subset=["end", "val"])
        if fy_rows.empty:
            continue
        filled_fy = 0
        for _, r in fy_rows.iterrows():
            end = pd.to_datetime(r["end"])
            val = float(r["val"]) if np.isfinite(r["val"]) else np.nan
            if not np.isfinite(val):
                continue
            # FY ending in December → Q4 slot
            if end.month == 12:
                mask = (panel["date"] == end) & (panel["period_type"] == "Q4")
            else:
                # Non-Dec fiscal year end: match the quarter
                q_map = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}
                pt = q_map.get(end.month)
                if not pt:
                    continue
                mask = (panel["date"] == end) & (panel["period_type"] == pt)
            if mask.any() and panel.loc[mask, metric].isna().all():
                panel.loc[mask, metric] = val
                filled_fy += 1
        if filled_fy:
            logger.info("  ✓ Populated %s with %d FY balance-sheet values", metric, filled_fy)

    # Balance sheet snapping / fills
    for bs_col in [
        "cash", "current_assets", "net_ppe", "accumulated_depreciation", "gross_ppe",
        "intangibles", "total_assets", "current_liabilities", "long_term_debt",
        "equity", "credit_portfolio",
    ]:
        if bs_col in panel.columns:
            snap_balance_sheet(panel, bs_col)

    # Derived: gross_ppe (fill gaps where direct tag data is missing)
    if "net_ppe" in panel.columns:
        if "gross_ppe" not in panel.columns:
            panel["gross_ppe"] = np.nan
        panel["gross_ppe"] = pd.to_numeric(panel["gross_ppe"], errors="coerce")
        accum = pd.to_numeric(panel.get("accumulated_depreciation"), errors="coerce").fillna(0.0) if "accumulated_depreciation" in panel.columns else 0.0
        derived = pd.to_numeric(panel["net_ppe"], errors="coerce").fillna(0.0) + accum
        panel["gross_ppe"] = panel["gross_ppe"].fillna(derived)
        logger.info("  ✓ Filled gross_ppe gaps from net_ppe + accumulated_depreciation")

    # Derived: short_term_debt estimate if totally missing
    if "short_term_debt" in panel.columns and "long_term_debt" in panel.columns:
        if panel["short_term_debt"].isna().all():
            panel["short_term_debt"] = 0.15 * pd.to_numeric(panel["long_term_debt"], errors="coerce")
            logger.info("  ✓ Estimated short_term_debt as 15%% of long_term_debt")

    # Derived: taxes_paid fallback from current_tax_expense or tax_expense
    if "taxes_paid" in panel.columns:
        missing_pct = panel["taxes_paid"].isna().mean()
        if missing_pct > 0.3:  # More than 30% missing
            if "current_tax_expense" in panel.columns:
                filled = panel["taxes_paid"].isna() & panel["current_tax_expense"].notna()
                panel.loc[filled, "taxes_paid"] = panel.loc[filled, "current_tax_expense"]
                if filled.sum():
                    logger.info("  ✓ Filled %d taxes_paid gaps from current_tax_expense", int(filled.sum()))
            if "tax_expense" in panel.columns:
                still_missing = panel["taxes_paid"].isna() & panel["tax_expense"].notna()
                panel.loc[still_missing, "taxes_paid"] = (
                    pd.to_numeric(panel.loc[still_missing, "tax_expense"], errors="coerce") * 0.85
                )
                if still_missing.sum():
                    logger.info("  ✓ Filled %d taxes_paid gaps from tax_expense × 0.85", int(still_missing.sum()))

    # Q4 inference using FY facts (robust)
    if cfg.infer_q4:
        logger.info("Creating quarterly dataset with proper alignment...")
        inferred_total = 0
        for metric, df in metric_facts.items():
            if metric in panel.columns:
                panel, added = infer_q4_for_metric(panel, df, metric)
                inferred_total += added
        if inferred_total:
            logger.info("  ✓ Inferred Q4 for %d metric-years using FY - (Q1+Q2+Q3) [ROBUST]", inferred_total)

    # Keep most recent quarters if requested
    panel = panel.sort_values("date").reset_index(drop=True)
    if cfg.max_quarters is not None and len(panel) > cfg.max_quarters:
        panel = panel.tail(cfg.max_quarters).reset_index(drop=True)

    # Add ticker
    panel["ticker"] = cfg.ticker

    # Column order
    front = ["date", "period_type", "ticker"]
    rest = [c for c in panel.columns if c not in front]
    panel = panel[front + rest]

    # Quality scoring
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 80)

    def cov(col: str) -> float:
        if col not in panel.columns:
            return 0.0
        return float(panel[col].notna().mean()) * 100.0

    key_cols = [
        "revenue", "ebit", "net_income", "operating_cf",
        "net_ppe", "accumulated_depreciation", "long_term_debt",
        "credit_portfolio", "taxes_paid"
    ]

    scores = []
    for c in key_cols:
        if c in panel.columns:
            pct = cov(c)
            scores.append(pct)
            flag = "✓✓" if pct >= 95 else ("✓" if pct >= 75 else ("⚠" if pct >= 50 else "✗"))
            logger.info("  %s %-26s : %5.1f%%", flag, c, pct)

    overall = float(np.mean(scores)) if scores else 0.0
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL DATA QUALITY SCORE")
    logger.info("=" * 80)
    logger.info("Overall: %.1f%%", overall)

    # Save
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cfg.output_csv, index=False)

    logger.info("\n✓ HIGH-QUALITY DATA SAVED: %s", cfg.output_csv)
    logger.info("  Quarters: %d", len(panel))
    logger.info("  Date range: %s to %s", panel["date"].min(), panel["date"].max())
    logger.info("  Columns: %d", len(panel.columns))
    logger.info("  Overall quality: %.1f%%", overall)
    logger.info("=" * 80)

    print("\n" + "=" * 80)
    print("✓✓ SUCCESS - HIGH-QUALITY DATA EXTRACTED")
    print("=" * 80)
    print(f"\nDataset: {len(panel)} quarters with {len(panel.columns)-3} metrics")
    print(f"Coverage: {panel['date'].min().date()} to {panel['date'].max().date()}\n")
    print("Recent quarters (last 5):")
    show_cols = [c for c in ["date", "period_type", "revenue", "ebit", "current_assets", "net_ppe"] if c in panel.columns]
    print(panel[show_cols].tail(5).to_string(index=False))
    print("\n" + "=" * 80)
    print("Ready to run: python src/main.py")
    print("=" * 80)


if __name__ == "__main__":
    main()