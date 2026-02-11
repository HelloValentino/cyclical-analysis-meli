# Data Quality: Grouping, Classification & Preparation

> How raw SEC filings and macro series become the institutional-grade master dataset.

---

## Table of Contents

1. [End-to-End Data Flow](#1-end-to-end-data-flow)
2. [SEC XBRL Extraction & Tag Mapping](#2-sec-xbrl-extraction--tag-mapping)
3. [Quarterly Alignment & Deduplication](#3-quarterly-alignment--deduplication)
4. [YTD-to-Quarterly Conversion](#4-ytd-to-quarterly-conversion)
5. [FY → Q4 Population (Balance Sheet)](#5-fy--q4-population-balance-sheet)
6. [Forward-Fill & Snapping](#6-forward-fill--snapping)
7. [Q4 Inference](#7-q4-inference)
8. [Derived Fields & Fallbacks](#8-derived-fields--fallbacks)
9. [Macro Data Acquisition & Tagging](#9-macro-data-acquisition--tagging)
10. [VP Metric Calculations](#10-vp-metric-calculations)
11. [Regime Classification](#11-regime-classification)
12. [Phase Space Quadrant Tagging](#12-phase-space-quadrant-tagging)
13. [Cycle Decomposition](#13-cycle-decomposition)
14. [Log Transforms & S-Curve Detection](#14-log-transforms--s-curve-detection)
15. [Data Quality Scoring](#15-data-quality-scoring)

---

## 1. End-to-End Data Flow

```
SEC EDGAR (Company Facts API)
        │
        ▼
┌──────────────────────────┐
│  Raw XBRL Facts (JSON)   │  ← cached in data/raw/sec_cache/
│  per tag, per filing     │
└──────────┬───────────────┘
           │  extract + dedup per (end, fp)
           ▼
┌──────────────────────────┐
│  Metric Fact Tables      │  ← one per financial metric
│  (tag, start, end, fp,   │
│   form, filed, val)      │
└──────────┬───────────────┘
           │  populate quarterly panel
           │  + YTD→quarterly conversion
           │  + FY→Q4 population
           │  + forward-fill (balance sheet)
           │  + Q4 inference (FY - Q1-Q3)
           ▼
┌──────────────────────────┐
│  Quarterly Financial     │  ← data/raw/company/meli_financials.csv
│  Panel (40 quarters)     │    columns: date, period_type, ticker, 25 metrics
└──────────┬───────────────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
┌────────┐  ┌────────────────┐
│ VP     │  │ Macro Data     │  ← BCB SGS API (CPI, SELIC, USD/BRL)
│ Calcs  │  │ (daily→qtrly)  │    data/raw/macro/macro_master.csv
└───┬────┘  └───────┬────────┘
    │               │
    │  Advanced analytics (regime, cycle, phase space)
    │               │
    ▼               ▼
┌──────────────────────────┐
│  Master Dataset          │  ← data/processed/master_dataset.parquet
│  (40 × ~146 columns)    │    company + macro + VP metrics + tags
└──────────────────────────┘
           │
           ▼
    Visualizations + Summary Tables
    (outputs/figures/, outputs/tables/)
```

---

## 2. SEC XBRL Extraction & Tag Mapping

### Source

All company financial data comes from the **SEC EDGAR Company Facts API**:

```
https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
```

This returns every XBRL fact ever filed by the company across all 10-Q and 10-K
filings, organized by US-GAAP taxonomy tag.

### Multi-Tag Strategy

A single financial concept (e.g. "long-term debt") may be reported under different
XBRL tags across filing periods. The extractor defines **ordered alternative tags**
per metric, with earlier tags having higher priority:

| Metric | Primary Tag | Alternative Tags |
|---|---|---|
| `revenue` | `RevenueFromContractWithCustomerExcludingAssessedTax` | `Revenues`, `SalesRevenueServicesNet` |
| `accumulated_depreciation` | `AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment` | `...FinanceLeaseRightOfUseAssetAccumulatedDepreciationAndAmortization` |
| `gross_ppe` | `PropertyPlantAndEquipmentGross` | `...FinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization` |
| `long_term_debt` | `LongTermDebt` | `LongTermDebtAndCapitalLeaseObligations`, `LongTermLoansPayable` |
| `credit_portfolio` | `NotesReceivableNet` | `LoansAndLeasesReceivableNetReportedAmount`, `NotesAndLoansReceivableNetCurrent`, `FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss` |
| `short_term_debt` | `DebtCurrent` | `LongTermDebtCurrent`, `ShortTermBorrowings` |

### Merge Logic (`merge_all_tags`)

Facts from all available tags are combined into a single table. When multiple tags
report data for the same `(end_date, fiscal_period)` pair, the **first-listed tag wins**
(priority ordering). Within a single tag, deduplication favors:

1. **Form rank**: 10-Q (rank 3) > 10-K (rank 2) > other (rank 0)
2. **Filing date**: latest filed wins (most recent restatement)

This ensures the most authoritative, most recent value is always selected.

### Outlier Guard (Post-Merge Discontinuity Detection)

After priority-based deduplication, an **outlier guard** scans the merged series
for values that appear to have been filed in the wrong unit (a known SEC XBRL
filing error where values are reported in thousands instead of units, producing
~1000× discontinuities).

**Algorithm — Log-space jump-and-recover detection:**

1. Convert the merged value series to log₁₀ space (so multiplicative
   discontinuities become additive jumps).
2. Compute first-differences of the log series.
3. Identify **big jumps** — differences exceeding `log₁₀(50) ≈ 1.7` in
   absolute value. Each jump is classified as a *drop* (negative) or *rise*
   (positive).
4. Pair consecutive **opposite** jumps: a *drop* followed by a *rise* marks a
   **dip-and-recover** segment; a *rise* followed by a *drop* marks a
   **spike-and-revert** segment. All values between paired jumps are flagged as
   outliers.
5. Same-direction consecutive jumps (e.g. two drops) are treated as genuine
   level shifts and are **not** flagged.

**Why not a rolling median?** A naïve rolling-median approach fails when there
are **consecutive** corrupted values, because the median itself becomes
corrupted. The jump-and-recover approach tracks *transitions* between magnitude
levels rather than comparing to a local statistic, making it robust to runs of
bad values of any length.

**Replacement:** For each flagged value, the system substitutes the next-priority
tag's value for the same `(end_date, fiscal_period)` slot. If no alternative
exists, the value is left as-is.

**Example — credit_portfolio Q2/Q3 2022:**

| Quarter | Original (bad) | Replaced with | Source tag |
|---|---|---|---|
| 2021-12-31 Q2 | $1,260,000 | $1,199,000,000 | `NotesAndLoansReceivableNetCurrent` |
| 2021-12-31 Q3 | $1,260,000 | $1,199,000,000 | `NotesAndLoansReceivableNetCurrent` |
| 2022-06-30 Q2 | $1,845,000 | $1,790,000,000 | `NotesAndLoansReceivableNetCurrent` |
| 2022-09-30 Q3 | $1,766,000 | $1,724,000,000 | `NotesAndLoansReceivableNetCurrent` |

These values from `LoansAndLeasesReceivableNetReportedAmount` were filed in
thousands instead of units. The guard detected the dip-and-recover pattern and
automatically substituted the correct values from the next-priority tag.

---

## 3. Quarterly Alignment & Deduplication

### Quarter Index Construction

A uniform quarterly grid is built from the earliest to latest quarter-end dates
observed across all metrics:

```
date         period_type
2015-12-31   Q4
2016-03-31   Q1
2016-06-30   Q2
...
2025-09-30   Q3
```

### Populating the Panel

Each metric's quarterly facts (fp ∈ {Q1, Q2, Q3, Q4}) are placed into the panel
by matching `(end_date, period_type)`. Only facts where `fp` matches the expected
quarter are accepted.

### Deduplication Rule

Per `(end_date, fiscal_period)`, one record survives:
- **Best form** first (10-Q preferred for quarterly data)
- **Latest filing date** as tiebreaker (picks up restatements)

---

## 4. YTD-to-Quarterly Conversion

### The Problem

SEC EDGAR reports **cash flow statement items** as **year-to-date (YTD) cumulative**
values within each fiscal year:

| Period | Raw Value | Meaning |
|---|---|---|
| Q1 | 100 | Jan–Mar (standalone) |
| Q2 | 250 | Jan–Jun (cumulative, not Apr–Jun) |
| Q3 | 400 | Jan–Sep (cumulative, not Jul–Sep) |

If these are placed directly into a quarterly panel, Q2 and Q3 appear artificially
inflated, and Q4 inference (FY − Q1 − Q2 − Q3) produces nonsense.

### Detection

The extractor detects YTD patterns by checking whether Q2/Q3 facts have a `start`
date of January 1 (or January 2). If ≥50% of Q2/Q3 entries start in January,
the metric is flagged as YTD.

### Conversion

For each calendar year:

```
Q1_standalone = Q1_raw                    (already standalone)
Q2_standalone = Q2_raw − Q1_raw           (Apr–Jun)
Q3_standalone = Q3_raw − Q2_raw           (Jul–Sep)
Q4              inferred later via FY − (Q1+Q2+Q3) on standalone values
```

### Affected Metrics

| Metric | Description |
|---|---|
| `taxes_paid` | `IncomeTaxesPaid` — cash taxes paid |
| `operating_cf` | `NetCashProvidedByUsedInOperatingActivities` |
| `ppe_additions` | `PaymentsToAcquirePropertyPlantAndEquipment` |
| `intangible_additions` | `PaymentsToAcquireIntangibleAssets` |

---

## 5. FY → Q4 Population (Balance Sheet)

### The Problem

Some XBRL tags only report annual (`fp=FY`) data with no quarterly breakdowns. For
**balance sheet items** (point-in-time snapshots), the FY-end value *is* the Q4 value
because both refer to the same date (e.g., December 31).

### Logic

For each FY row where:
- The end date falls in **December** → populate the **Q4** slot for that date
- The end date falls in another month (non-Dec fiscal year) → map to the
  corresponding quarter (Mar→Q1, Jun→Q2, Sep→Q3)

Only fills empty slots — does not overwrite existing quarterly data.

### Affected Metrics (Balance Sheet Group)

`cash`, `current_assets`, `net_ppe`, `accumulated_depreciation`, `gross_ppe`,
`intangibles`, `total_assets`, `current_liabilities`, `long_term_debt`,
`equity`, `credit_portfolio`

---

## 6. Forward-Fill & Snapping

### Balance Sheet Snapping

Balance sheet items are **point-in-time** (the value at a given date persists until
the next reporting date). After populating quarterly and FY→Q4 values, the
extractor applies **forward-fill (`ffill`)** to propagate the last known value into
subsequent empty quarters.

This is appropriate because balance sheet values don't "expire" — if a company
reported $1B in net PP&E as of Q2, that value remains the best estimate for Q3
until the Q3 filing arrives.

### Columns Affected

All balance sheet metrics listed in §5 above receive forward-fill treatment.

> **Note**: Flow metrics (revenue, EBIT, cash flows) are **never** forward-filled.
> A missing quarter for a flow metric stays `NaN` rather than carrying forward a
> stale value.

---

## 7. Q4 Inference

### Method

When Q4 data is missing for a metric but FY and Q1–Q3 are available:

```
Q4 = FY − (Q1 + Q2 + Q3)
```

### Safeguards

| Check | Purpose |
|---|---|
| **YTD detection** | If Q1 < Q2 < Q3 monotonically (by >15%, >8%), values are converted to standalone before subtraction |
| **Non-negative guard** | For flow metrics (revenue, capex, taxes…), inferred Q4 must be ≥ 0 |
| **Outlier guard** | Inferred Q4 must not exceed 10× the median of Q1–Q3 |
| **Short-span FY reclassification** | FY entries with span ≤100 days ending in December are reclassified as Q4 directly (these are quarterly data mis-labeled as FY) |
| **End-month filtering** | Quarter facts must match expected end-month (Q1→Mar, Q2→Jun, Q3→Sep, Q4→Dec) to prevent cross-contamination from comparative-period data |

---

## 8. Derived Fields & Fallbacks

### Gross PP&E

```
gross_ppe = net_ppe + accumulated_depreciation
```

Used when the direct `PropertyPlantAndEquipmentGross` tag has gaps.

### Short-Term Debt Estimate

If `short_term_debt` is entirely missing:

```
short_term_debt = 0.15 × long_term_debt
```

### Taxes Paid Fallback Chain

When `taxes_paid` has >30% missing values:

1. **First fallback**: fill from `current_tax_expense` (current portion of income tax)
2. **Second fallback**: fill from `tax_expense × 0.85` (total tax × cash realization factor)

---

## 9. Macro Data Acquisition & Tagging

### Source: Brazil Central Bank (BCB SGS API)

| Series | ID | Frequency | Description |
|---|---|---|---|
| CPI (IPCA) | 433 | Monthly | Consumer price index |
| SELIC | 432 | Daily | Policy interest rate |
| USD/BRL | 1 | Daily | Exchange rate |

Daily series are fetched in ≤9-year windows (BCB enforces a 10-year API limit),
then concatenated.

### Derived Macro Metrics

| Metric | Formula | Meaning |
|---|---|---|
| `selic_vol_60d` | 60-day rolling std of daily SELIC changes | Policy rate volatility |
| `usd_brl_vol_12m` | 252-day rolling std of log-returns × √252 | Annualized FX volatility |
| `cpi_yoy` | 12-month % change in CPI index | Year-over-year inflation |
| `real_rate` | SELIC (decimal) − CPI YoY | Real interest rate |

### Tagging

Every macro observation is tagged with:
- **`country`**: `"BRA"` (Brazil)
- **`date`**: observation date (daily or monthly)

CPI and `cpi_yoy` are forward-filled onto the daily grid so that every business
day has a value.

### Quarterly Aggregation

Before merging with the company panel, macro data is aggregated to quarterly
frequency by taking the **mean** of each numeric column per `(quarter, country)`.
Categorical columns (e.g., regime labels) take the **first** observed value per quarter.

---

## 10. VP Metric Calculations

After the quarterly panel is loaded, the `VPCalculator` computes VP Capital Cycle
metrics. Key groupings:

### Invested Capital (Economic Definition)

```
NWC             = (Current Assets − Cash) − (Current Liabilities − Short-Term Debt)
Capitalized R&D = rolling 16-quarter sum of R&D expense ÷ amortization life (4 yr)
Invested Capital = NWC + Gross PP&E + Capitalized R&D
```

This uses **gross** PP&E (not net) and capitalizes R&D, consistent with VP/HOLT
economic capital methodology.

### Return Metrics

| Metric | Formula |
|---|---|
| `cash_tax_rate` | taxes_paid / EBIT (fallback: tax_expense / EBIT, then 25%) |
| `nopat` | EBIT × (1 − cash_tax_rate) + R&D expense − R&D amortization |
| `nopat_ttm` | trailing 4-quarter sum of NOPAT |
| `roic` | NOPAT ÷ average Invested Capital |
| `roic_ttm` | NOPAT TTM ÷ average Invested Capital |
| `economic_profit` | Invested Capital × (ROIC − WACC) |
| `incremental_roic` | ΔNOPAT(3yr) ÷ ΔIC(3yr) |

### WACC Tagging

```
WACC = base_wacc (12%) + country_risk_premium (BRA: 4%) = 16%
```

Each row is tagged with the appropriate WACC based on its `country` field.

### Capital Intensity Metrics

| Metric | Formula |
|---|---|
| `maintenance_capex` | Depreciation & amortization |
| `growth_capex` | (PPE additions + Intangible additions) − maintenance capex |
| `total_capex_intensity` | Total capex TTM ÷ Revenue TTM |
| `growth_opex_intensity` | (Sales & Marketing + R&D) TTM ÷ Revenue TTM |
| `reinvestment_rate` | Total reinvestment ÷ NOPAT TTM |

### VP/HOLT Adjusted ROIC

For platform companies, the adjusted ROIC capitalizes additional intangible
investments as economic assets:

| Capitalized Asset | Input Series | Amortization Life |
|---|---|---|
| Sales & Marketing | `sales_marketing` | 8 quarters (2 years) |
| Software / Intangibles | `intangible_additions` | 16 quarters (4 years) |
| R&D | `rd_expense` | 16 quarters (4 years) |

```
Adjusted IC = Gross PP&E + Capitalized R&D + Capitalized S&M
            + Capitalized Software + NWC + Operating Credit Assets
Adjusted ROIC = Adjusted NOPAT TTM ÷ Average Adjusted IC
```

---

## 11. Regime Classification

### Continuous Capital Scarcity Index

The macro environment is classified using a **continuous scarcity index** (0–100)
rather than binary regimes. This captures the spectrum between capital abundance
and capital scarcity.

#### Component Signals

| Component | Weight | Source | Interpretation |
|---|---|---|---|
| `inflation_vol` | 30% | 12-month rolling std of CPI | Higher → more scarcity |
| `fx_vol` | 25% | Annualized FX volatility | Higher → more scarcity |
| `real_rate_repression` | 25% | −real_rate | More negative real rates → more scarcity |
| `credit_collapse` | 20% | −rolling mean of CPI (proxy) | Falling credit → more scarcity |

#### Normalization

Each component is **z-score normalized** (mean=0, std=1), then combined as a
weighted sum. The composite index is then rescaled to 0–100 using the **5th and
95th percentile** as bounds:

```
scarcity_normalized = (raw − P5) / (P95 − P5) × 100
```

Clamped to [0, 100]. If the data is degenerate (constant or missing), defaults
to **50** (neutral).

### Regime Probability

The scarcity index is converted to a probability via **logistic transformation**:

```
P(scarcity) = 1 / (1 + exp(−k × scarcity_index_continuous))
```

Where `k = 2.0` (steepness parameter).

### Regime Labels

The probability is discretized into four labeled regimes:

| Label | Probability Range | Interpretation |
|---|---|---|
| **abundant** | 0% – 25% | Capital freely available; low returns expected |
| **normal** | 25% – 50% | Balanced macro conditions |
| **scarce** | 50% – 75% | Capital becoming scarce; supports higher ROIC |
| **severe** | 75% – 100% | Extreme scarcity; crisis-like conditions |

Each row in the master dataset carries both the continuous probability
(`prob_scarcity`) and the discrete label (`regime_probabilistic`).

---

## 12. Phase Space Quadrant Tagging

### Concept

The phase space plots **returns (ROIC) vs. investment intensity** to identify where
a company sits in the capital cycle. Both axes are **z-score normalized** to remove
scale effects and center on zero.

### Axes

| Axis | Variable | Z-Score Column |
|---|---|---|
| Y (vertical) | `roic_ttm` | `roic_ttm_z` |
| X (horizontal) | `total_growth_investment_intensity` (preferred) or `total_capex_intensity` (fallback) | `tgi_z` or `total_capex_intensity_z` |

### Quadrant Classification

```
              High ROIC (y ≥ 0)
                    │
    Value           │         Super
    Creation        │         Cycle
   (high return,    │    (high return,
    low invest)     │     high invest)
                    │
  ──────────────────┼──────────────────
                    │
    Capital         │        Capital
    Abundance       │        Destruction
   (low return,     │    (low return,
    low invest)     │     high invest)
                    │
              Low ROIC (y < 0)
```

| Quadrant | Condition | Interpretation |
|---|---|---|
| **super_cycle** | y ≥ 0, x ≥ 0 | Rare: high returns AND high investment. VP theory predicts mean reversion. |
| **value_creation** | y ≥ 0, x < 0 | High returns with declining investment — classic value-creating phase. |
| **capital_abundance** | y < 0, x < 0 | Low returns, low investment — trough of cycle, potential recovery ahead. |
| **capital_destruction** | y < 0, x ≥ 0 | Low returns despite high investment — capital being destroyed. |

### Super-Cycle Signal

A special flag is raised when ALL of the following hold:
- Phase quadrant = `super_cycle`
- ROIC trend > 20%
- Probability of scarcity > 50%

This combination is extremely rare and historically signals a structural shift.

---

## 13. Cycle Decomposition

### Hodrick-Prescott (HP) Filter

Each key metric is decomposed into a **structural trend** and a **cyclical component**
using the HP filter with λ=1600 (standard for quarterly data):

```
observed = trend + cycle
```

### Decomposed Metrics

| Input Metric | Trend Column | Cycle Column |
|---|---|---|
| `roic_ttm` | `roic_trend` | `roic_cycle` |
| `total_capex_intensity` | `total_capex_intensity_trend` | `total_capex_intensity_cycle` |
| `growth_opex_intensity` | `growth_opex_intensity_trend` | `growth_opex_intensity_cycle` |
| `economic_profit_ttm` | `economic_profit_ttm_trend` | `economic_profit_ttm_cycle` |
| `reinvestment_rate` | `reinvestment_rate_trend` | `reinvestment_rate_cycle` |
| `incremental_roic` | `incremental_roic_trend` | `incremental_roic_cycle` |

Requires a minimum of 8 non-null observations to run (otherwise skipped).

---

## 14. Log Transforms & S-Curve Detection

### Purpose

Exponentially growing metrics (credit portfolio, revenue) follow S-curves. Log
transforms linearize the growth phase, making **inflection points** visible as
sign changes in the acceleration term.

### Transform Pipeline

For each metric in `{credit_portfolio, economic_profit_ttm, revenue, invested_capital, total_assets}`:

```
log_value    = log(1 + max(0, value))           # log1p, clipped non-negative
log_growth   = Δ(log_value) × 100               # quarter-over-quarter log growth (%)
log_accel    = Δ(log_growth)                     # acceleration (2nd derivative)
```

### Inflection Detection

- **`log_accel > 0`** → pre-inflection (growth accelerating)
- **`log_accel < 0`** → post-inflection (growth decelerating, S-curve flattening)

This is used in the institutional summary to tag `credit_inflection_status`.

---

## 15. Data Quality Scoring

### Key Fields Monitored

The extractor validates coverage for these critical fields:

| Field | Expected Coverage | Role |
|---|---|---|
| `revenue` | ≥95% | Core top-line metric |
| `ebit` | ≥95% | Operating profitability |
| `net_income` | ≥95% | Bottom-line profitability |
| `operating_cf` | ≥75% | Cash flow generation |
| `net_ppe` | ≥95% | Capital base |
| `accumulated_depreciation` | ≥75% | Gross PP&E derivation |
| `long_term_debt` | ≥75% | Capital structure |
| `credit_portfolio` | ≥75% | Fintech segment tracking |
| `taxes_paid` | ≥75% | Cash tax rate accuracy |

### Scoring Thresholds

| Symbol | Coverage | Meaning |
|---|---|---|
| ✓✓ | ≥ 95% | Excellent — institutional grade |
| ✓ | ≥ 75% | Good — acceptable for analysis |
| ⚠ | ≥ 50% | Warning — results may be unreliable |
| ✗ | < 50% | Critical — field effectively unusable |

### Overall Score

The overall data quality score is the **arithmetic mean** of coverage percentages
across all key fields. Target: **≥ 90%** for institutional-grade analysis.

### Quality Improvement Log

| Field | Before | After | Fix Applied |
|---|---|---|---|
| `accumulated_depreciation` | 0% | 100% | Added lease-era XBRL tag + FY→Q4 population + forward-fill |
| `taxes_paid` | 0% | 75% | YTD→quarterly conversion + fallback from tax_expense |
| `long_term_debt` | 32.5% | 100% | Added 2 alternative XBRL tags with broader coverage |
| `credit_portfolio` | 17.5% | 90% | Added 3 alternative XBRL tags spanning different eras |
| **Overall** | **57.5%** | **92.5%** | **Multi-tag, YTD conversion, FY→Q4, fallbacks** |

---

## Column Tags in the Master Dataset

Every row in the final `master_dataset.parquet` carries these classification tags:

| Tag Column | Example Value | Source |
|---|---|---|
| `ticker` | `MELI` | SEC extraction config |
| `country` | `BRA` | Company default / macro data |
| `date` | `2024-09-30` | Quarter-end date |
| `period_type` | `Q3` | Fiscal period (Q1/Q2/Q3/Q4) |
| `regime_probabilistic` | `normal` | Macro regime label (§11) |
| `prob_scarcity` | `0.52` | Continuous scarcity probability (§11) |
| `scarcity_index_normalized` | `45.2` | Scarcity index 0–100 (§11) |
| `phase_quadrant` | `capital_abundance` | Capital cycle phase (§12) |
| `roic_trend` | `0.18` | HP-filtered structural ROIC (§13) |
| `roic_cycle` | `-0.02` | Cyclical deviation from trend (§13) |
| `credit_portfolio_log_accel` | `-1.3` | S-curve inflection signal (§14) |

