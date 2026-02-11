# src/data_acquisition.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ----------------------------
# Company loader
# ----------------------------
@dataclass
class CompanyDataLoader:
    data_dir: str

    def load_financials(
        self,
        csv_path: str,
        default_ticker: str = "MELI",
        default_country: str = "BRA",
    ) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Financial data not found at: {path}")

        df = pd.read_csv(path)

        for col in ["date", "period_type"]:
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' in {path.name}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df["period_type"] = df["period_type"].astype(str).str.upper().str.strip()

        if "ticker" not in df.columns:
            logger.warning("Financials missing 'ticker' column. Injecting '%s'.", default_ticker)
            df["ticker"] = default_ticker
        else:
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df.loc[df["ticker"].isna() | (df["ticker"] == ""), "ticker"] = default_ticker

        if "country" not in df.columns:
            df["country"] = default_country
        else:
            df["country"] = df["country"].astype(str).str.upper().str.strip()
            df.loc[df["country"].isna() | (df["country"] == ""), "country"] = default_country

        # Keep last per (ticker,date) to handle inferred/restated rows without hard-crash
        if df.duplicated(subset=["ticker", "date"]).any():
            dups = df[df.duplicated(subset=["ticker", "date"], keep=False)].sort_values(["ticker", "date"])
            logger.warning(
                "Duplicate (ticker,date) rows detected. Keeping last occurrence. Example duplicates:\n%s",
                dups[["ticker", "date", "period_type"]].head(20).to_string(index=False),
            )
            df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info("✓ Loaded %d periods", len(df))
        logger.info("  Date range: %s to %s", df["date"].min(), df["date"].max())
        logger.info("  Ticker: %s", df["ticker"].unique().tolist())
        logger.info("  Country: %s", default_country)

        return df


# ----------------------------
# Macro fetcher (Brazil BCB/SGS)
# ----------------------------

# Daily series IDs that require date-bounded requests (max 10-year window)
_DAILY_SERIES: set = {432, 1}  # SELIC, USD/BRL


def _sgs_url(series_id: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Build BCB SGS URL.  For daily series the API *requires* dataInicial and
    enforces a max 10-year window.  Dates must be dd/mm/yyyy.
    """
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
    if start_date:
        url += f"&dataInicial={start_date}"
    if end_date:
        url += f"&dataFinal={end_date}"
    return url


def _date_chunks(start: str, end: str, max_years: int = 9) -> List[tuple]:
    """
    Split a YYYY-MM-DD date range into sub-ranges of at most *max_years* years
    (we use 9 to stay safely under BCB's 10-year cap).
    Returns list of (start_dd/mm/yyyy, end_dd/mm/yyyy) tuples.
    """
    from datetime import datetime, timedelta
    fmt_iso = "%Y-%m-%d"
    fmt_bcb = "%d/%m/%Y"
    s = datetime.strptime(start, fmt_iso)
    e = datetime.strptime(end, fmt_iso)
    chunks = []
    while s < e:
        chunk_end = s.replace(year=s.year + max_years)
        if chunk_end > e:
            chunk_end = e
        chunks.append((s.strftime(fmt_bcb), chunk_end.strftime(fmt_bcb)))
        s = chunk_end + timedelta(days=1)
    return chunks


def _robust_json_get(
    url: str,
    timeout: int = 30,
    max_retries: int = 3,
    backoff: float = 1.35,
) -> list:
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": "VP-Capital-Cycle/1.0",
    }

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            text = r.text or ""

            # BCB sometimes returns HTML/XML even with 200
            if text.lstrip().startswith("<"):
                raise RuntimeError(f"Non-JSON payload returned. Starts: {text[:120]!r}")

            payload = json.loads(text)
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected JSON payload type: {type(payload)}")

            return payload

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff ** (attempt - 1))
                continue

    raise RuntimeError(f"Failed to fetch valid SGS JSON after retries. Last error: {last_err}") from last_err


def _payload_to_df(payload: list, series_name: str) -> pd.DataFrame:
    df = pd.DataFrame(payload)
    if df.empty:
        return df

    # SGS uses dd/mm/yyyy
    df["date"] = pd.to_datetime(df.get("data"), errors="coerce", dayfirst=True)
    df["value"] = pd.to_numeric(df.get("valor"), errors="coerce")

    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    df["series"] = series_name
    df["country"] = "BRA"
    return df[["date", "country", "series", "value"]]


@dataclass
class MacroDataFetcher:
    start_date: str  # "YYYY-MM-DD"
    end_date: str    # "YYYY-MM-DD"
    output_dir: str
    timeout: int = 30

    def fetch_brazil(self) -> pd.DataFrame:
        """
        Brazil macro via BCB SGS:
        - IPCA CPI (monthly): 433
        - SELIC (daily): 432
        - USD/BRL (daily): 1

        Daily series require dataInicial/dataFinal and enforce a max 10-year
        window, so we split those into <=9-year chunks and concatenate.
        Monthly series (CPI) work without date params.
        """
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Fetching Brazil macro data from BCB (SGS)...")

        series_map: Dict[str, int] = {
            "cpi": 433,
            "selic": 432,
            "usd_brl": 1,
        }

        frames: List[pd.DataFrame] = []
        for name, sid in series_map.items():
            try:
                if sid in _DAILY_SERIES:
                    # Daily series: chunk into <=9-year windows
                    chunks = _date_chunks(self.start_date, self.end_date, max_years=9)
                    chunk_frames: List[pd.DataFrame] = []
                    for s_bcb, e_bcb in chunks:
                        url = _sgs_url(sid, start_date=s_bcb, end_date=e_bcb)
                        payload = _robust_json_get(url, timeout=self.timeout, max_retries=3)
                        cdf = _payload_to_df(payload, series_name=name)
                        if not cdf.empty:
                            chunk_frames.append(cdf)
                    if chunk_frames:
                        df = pd.concat(chunk_frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                        frames.append(df)
                        logger.info("  ✓ Fetched %s: %d observations (%d chunk(s))", name, len(df), len(chunks))
                    else:
                        logger.warning("  ⚠ %s: no data returned across %d chunk(s)", name, len(chunks))
                else:
                    # Monthly series: no date params needed
                    url = _sgs_url(sid)
                    payload = _robust_json_get(url, timeout=self.timeout, max_retries=3)
                    df = _payload_to_df(payload, series_name=name)
                    if not df.empty:
                        frames.append(df)
                    logger.info("  ✓ Fetched %s: %d observations", name, len(df))
            except Exception as e:
                # keep pipeline alive; SELIC/FX strongly preferred but not fatal
                logger.error("  ✗ Failed to fetch %s: %s", name, e)

        if not frames:
            return pd.DataFrame()

        macro = pd.concat(frames, ignore_index=True).sort_values(["date", "series"]).reset_index(drop=True)

        # Filter locally to requested window
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        macro = macro[(macro["date"] >= start) & (macro["date"] <= end)].copy()

        brazil_path = out_dir / "brazil_macro.csv"
        macro.to_csv(brazil_path, index=False)
        logger.info("✓ Saved Brazil data: %d rows to %s", len(macro), brazil_path)

        return macro

    def calculate_volatility_metrics(self, macro: pd.DataFrame) -> pd.DataFrame:
        """
        Wide panel + derived metrics:
        - selic_vol_60d: rolling std of daily changes
        - usd_brl_vol_12m: rolling std of daily log-returns * sqrt(252)
        - cpi_yoy: YoY inflation from monthly CPI
        - real_rate: (selic% -> decimal) - cpi_yoy (ffilled)
        """
        if macro is None or macro.empty:
            return pd.DataFrame()

        pivot = macro.pivot_table(index="date", columns="series", values="value", aggfunc="last").sort_index()

        # SELIC vol (daily changes)
        if "selic" in pivot.columns and pivot["selic"].notna().any():
            s = pd.to_numeric(pivot["selic"], errors="coerce")
            ds = s.diff()
            pivot["selic_vol_60d"] = ds.rolling(60, min_periods=20).std()

        # FX vol (log returns)
        if "usd_brl" in pivot.columns and pivot["usd_brl"].notna().any():
            fx = pd.to_numeric(pivot["usd_brl"], errors="coerce")
            r = np.log(fx).diff()
            pivot["usd_brl_vol_12m"] = r.rolling(252, min_periods=120).std() * np.sqrt(252)

        # CPI YoY (monthly) — must compute on monthly-only slice, then merge back
        if "cpi" in pivot.columns and pivot["cpi"].notna().any():
            cpi_monthly = pivot["cpi"].dropna()
            cpi_yoy_monthly = cpi_monthly.pct_change(12)
            pivot["cpi_yoy"] = cpi_yoy_monthly  # NaN on non-CPI dates

        pivot = pivot.reset_index()
        pivot["country"] = "BRA"
        pivot = pivot.sort_values("date").reset_index(drop=True)

        # Forward fill CPI and CPI YoY onto the daily grid
        if "cpi" in pivot.columns:
            pivot["cpi"] = pivot["cpi"].ffill()
        if "cpi_yoy" in pivot.columns:
            pivot["cpi_yoy"] = pivot["cpi_yoy"].ffill()

        # real_rate = selic(decimal) - cpi_yoy(decimal)
        if "selic" in pivot.columns and "cpi_yoy" in pivot.columns:
            selic_dec = pd.to_numeric(pivot["selic"], errors="coerce") / 100.0
            cpi_yoy = pd.to_numeric(pivot["cpi_yoy"], errors="coerce")
            pivot["real_rate"] = selic_dec - cpi_yoy

        return pivot

    def fetch_all(self) -> pd.DataFrame:
        logger.info("\n" + "=" * 80)
        logger.info("FETCHING MACRO DATA")
        logger.info("=" * 80 + "\n")

        bra = self.fetch_brazil()
        if bra.empty:
            logger.warning("No Brazil data fetched")
            return pd.DataFrame()

        logger.info("Calculating macro volatility metrics...")
        master = self.calculate_volatility_metrics(bra)
        logger.info("✓ Volatility metrics calculated")

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        master_path = out_dir / "macro_master.csv"
        master.to_csv(master_path, index=False)

        logger.info("\n✓ Macro master dataset saved: %s", master_path)
        logger.info("  Total observations: %d", len(master))
        logger.info("  Date range: %s to %s", master["date"].min(), master["date"].max())
        logger.info("  Countries: %s", sorted(master["country"].unique().tolist()))

        return master


if __name__ == "__main__":
    print("data_acquisition.py loaded successfully (no standalone execution).")