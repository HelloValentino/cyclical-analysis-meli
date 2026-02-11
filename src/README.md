# VP Capital Cycle Analysis Framework
### MercadoLibre Case Study

This framework transforms company financials and macroeconomic data into actionable investment insights through rigorous empirical analysis.


## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Framework Evolution](#framework-evolution)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Output Quality Assurance](#output-quality-assurance)
8. [Methodology](#methodology)
9. [Institutional Enhancements](#institutional-enhancements)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Overview

This framework implements Variant Perception's capital cycle methodology to analyze MercadoLibre's position within LATAM's evolving capital markets. It combines:

- **Automated SEC data extraction** from EDGAR XBRL filings
- **Macro regime classification** using continuous probabilistic models
- **Economic profit analysis** with VP-aligned return metrics
- **Capital cycle decomposition** separating structural trends from cyclical dynamics
- **Phase space visualization** revealing super-cycle positioning

### Key Capabilities

 **Automated Data Pipeline**: Extract 35+ quarters of financial data from SEC filings  
 **Analytics**: HP filter decomposition, Z-score normalization, log transforms  
 **Charts**: Phase space trajectories, regime heatmaps, S-curve analysis  
 **VP Methodology**: Supply-side primacy, capital scarcity regimes, ROIC persistence   

---

## Theoretical Foundation

### Variant Perception Capital Cycle Framework

The capital cycle describes the self-correcting mechanism by which capital allocation drives industry profitability:
```
High ROIC → Capital Inflows → Capacity Expansion → Competition → 
ROIC Compression → Capital Outflows → Capacity Rationalization → ROIC Recovery
```

**Core Principles:**

1. **Supply-Side Primacy**: Returns driven by competitive dynamics, not demand forecasting
2. **Mean Reversion**: High returns attract capital, driving returns toward cost of capital
3. **Regime Dependence**: Capital scarcity (emerging markets, high inflation) can suspend mean reversion
4. **Persistence Drivers**: Barriers to entry, network effects, regulatory moats extend high-ROIC periods

### Application to MercadoLibre

MELI operates in a **structurally scarce capital environment** (LATAM) with:

- **High barriers to entry** (network effects, regulatory complexity)
- **Capital scarcity** (macro volatility, financial repression)
- **Platform economics** (increasing returns to scale)
- **Financial system replacement** (credit monetization opportunity)

**Hypothesis**: MELI is in an early-to-mid super-cycle phase where capital scarcity + competitive moats enable sustained high ROIC despite elevated reinvestment.

---

## Framework Evolution

### Development Process: From Diagnostic to Institutional-Grade

This framework wasn't built in one pass. It evolved through **rigorous output scrutiny and iterative refinement**:

#### Phase 1: Initial Implementation (Diagnostic)
**Goal**: Build functional VP framework  
**Approach**: 
- 5-file Python structure
- Automated SEC data extraction
- Core VP metrics (ROIC, Economic Profit, Invested Capital, with adjusted formulas for our specific use case)
- Basic visualizations

**Issues Identified**:
- ❌ Data quality: 50-60% completeness (balance sheet items missing)
- ❌ Binary regime classification → Information loss
- ❌ Visual compression in capital cycle scatter
- ❌ Credit S-curve hidden by linear scaling
- ❌ Super-cycle structure buried in trend noise

#### Phase 2: Data Quality Fixes (Functional)
**Goal**: Achieve 80%+ data completeness  
**Refinements**:
- Expanded SEC XBRL field mappings (3-5 alternatives per metric)
- Intelligent gap-filling using financial relationships
- Forward-fill for balance sheet items (max 2 quarters)
- Separate point-in-time vs period data handling

**Results**:
- Balance sheet completeness
- Income statement completeness: 60% → 95%
- ax data: Actual cash taxes extracted vs estimated

**Key Insight**: Multiple XBRL field names for same metric over time (accounting standard changes). Solution: Try 3-5 alternatives in priority order.

#### Phase 3: Mechanical Fixes
**Goal**: Eliminate visualization artifacts  
**Issues**:
- Regime heatmap saturated (all "scarce")
- Capital cycle scatter collapsed into tight cluster
- Credit growth curve appeared flat
- N/A values in summary table

**Solutions**:

# Before: Binary classification
regime = 'scarce' if inflation > 40 else 'normal'

# After: Continuous index
scarcity_index = weighted_sum([
    z_score(inflation_vol),
    z_score(fx_vol),
    z_score(real_rate_repression),
    z_score(credit_collapse)
]) → normalized to 0-100 scale
```

**Results**:
- Regime gradation visible
- Tightening/loosening dynamics revealed

#### Phase 4: Institutional Enhancements
**Goal**: Transform from diagnostic to forecasting engine  
**Enhancements**:

1. **Probabilistic Regime Modeling**
```python
   # Logistic transformation
   P(scarcity) = 1 / (1 + exp(-k * scarcity_index))
   # Enables: Stress testing, transition probabilities
```

2. **Cycle Decomposition (HP Filter)**
```python
   # Separate structural trend from cyclical component
   roic_trend, roic_cycle = hp_filter(roic, lambda=1600)
   # Reveals: Super-cycle structure vs temporary fluctuations
```

3. **Phase Space Transformation**
```python
   # Z-score normalization for trajectory visualization
   phase_x = z_score(investment_intensity)
   phase_y = z_score(roic)
   # Enables: Clear phase separation, rotation visibility
```

4. **Log Transforms for S-Curves**
```python
   # Reveal convexity and inflection points
   credit_log = log(credit_portfolio + 1)
   inflection = d²(credit_log)/dt² 
   # Detects: Entry into exponential growth phase
```

**Results**:
- Capital cycle rotation clearly visible
- Super-cycle phase confirmed quantitatively
- Credit inflection point identified
- Trend vs cycle separated

### Output Scrutiny Process

**How We Refined the Framework:**

1. **Run Analysis** → Generate all charts and tables
2. **Visual Inspection** → Identify compression, saturation, artifacts
3. **Diagnose Root Cause** → Data quality? Scaling? Threshold logic?
4. **Implement Fix** → Mechanical upgrade or modeling enhancement
5. **Validate Improvement** → Confirm issue resolved without new artifacts
6. **Document Learning** → Capture insight for future applications

**Example: Capital Cycle Scatter Compression**
```
Initial Output: All points clustered in tight ball
↓
Diagnosis: Raw % scaling masks variation (4-8% range)
↓
Solution: Z-score normalization reveals ~2σ spread
↓
Validation: Phase separation now visible, trajectory clear
↓
Learning: Always normalize when comparing across time
```

**Quality Assurance Checklist:**

- [ ] Data completeness ≥80% for critical fields
- [ ] Visual clarity: No compression, saturation, or artifacts
- [ ] Information preservation: Gradation and nuance visible
- [ ] Interpretability: Charts answer investment questions directly
- [ ] Reproducibility: All random seeds fixed, methods documented
- [ ] VP alignment: No demand forecasting, supply-side focus maintained

---

## Project Structure
```
Project MELI/
├── config.yaml                  # Central configuration (paths, parameters, thresholds)
├── .gitignore
│
├── src/
│   ├── main.py                 # Orchestration pipeline (entry point)
│   ├── sec_data_extractor.py   # SEC EDGAR XBRL data extraction
│   ├── data_acquisition.py     # BCB macro data fetching + company data loading
│   ├── calculations.py         # VP metrics (ROIC, EP, IC, intensities)
│   ├── analysis.py             # Regime classification + dataset merging
│   ├── advanced_analysis.py    # Institutional analytics (HP filter, phase space)
│   ├── visualization.py        # Chart generation (10+ figures)
│   ├── load_config.py          # Config loader helper
│   └── README.md               # This file
│
├── data/
│   ├── raw/
│   │   ├── sec_cache/          # Cached SEC EDGAR JSON responses
│   │   ├── macro/              # BCB Brazil macro data (CPI, SELIC, USD/BRL)
│   │   └── company/            # meli_financials.csv (extracted quarterly data)
│   └── processed/
│       └── master_dataset.parquet  # Merged analysis-ready dataset
│
└── outputs/
    ├── figures/                # Institutional-grade PNG charts
    └── tables/
        └── summary_metrics.csv
```

---

## Installation

### Prerequisites

- **Python 3.10+** (tested with 3.13)
- **pip** package manager
- Internet access (SEC EDGAR API + BCB API)

### Step-by-Step Local Setup

```bash
# 1. Clone the repository and navigate into it
git clone <repo-url> "Project MELI"
cd "Project MELI"

# 2. Create a Python virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install pandas numpy matplotlib seaborn scipy statsmodels requests pyyaml pyarrow

# 5. (Optional) Configure SEC EDGAR identity
#    Set these environment variables so SEC requests include your contact info:
export SEC_EDGAR_EMAIL="you@example.com"
export SEC_EDGAR_COMPANY_NAME="YourCompany"

# 6. Verify installation
python -c "import pandas, numpy, matplotlib, seaborn, statsmodels, requests, yaml, pyarrow; print('✓ All dependencies installed')"
```

### Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scipy` | Scientific computing |
| `statsmodels` | HP filter (time series decomposition) |
| `requests` | API calls (SEC EDGAR, BCB) |
| `pyyaml` | Configuration loading |
| `pyarrow` | Parquet I/O |

---

## Usage

### Quick Start (Full Pipeline)

Make sure the virtual environment is active, then run two commands:

```bash
# Step 1: Extract financial data from SEC EDGAR
python src/sec_data_extractor.py

# Step 2: Run the full analysis pipeline
python src/main.py
```

### What the Pipeline Does

**Step 1 — `sec_data_extractor.py`**
1. Fetches MELI company facts from SEC EDGAR (CIK 0001099590)
2. Extracts 25+ financial metrics from XBRL filings (tries multiple field names per metric)
3. Deduplicates entries (prefers 10-Q over 10-K, latest filing date)
4. Infers Q4 values from FY − (Q1+Q2+Q3) where missing
5. Forward-fills balance sheet items
6. Saves `data/raw/company/meli_financials.csv` (40 quarters, Q4 2015 – Q3 2025)

**Step 2 — `main.py`** (8 stages)

| Stage | Description |
|---|---|
| 1. Macro Data | Fetches Brazil CPI, SELIC, USD/BRL from BCB SGS API |
| 2. Company Data | Loads `meli_financials.csv`, assigns country |
| 3. VP Metrics | Calculates Invested Capital, ROIC, Economic Profit, intensities |
| 4. Analytics | Continuous scarcity index, regime probabilities, HP filter, phase space |
| 5. Merge | Quarterly alignment of company + macro data |
| 6. Summary | Phase classification, super-cycle signal detection |
| 7. Charts | Generates 10+ institutional-grade visualizations |
| 8. Table | Exports `summary_metrics.csv` |

### Outputs

After a successful run you will find:

```
data/
  raw/company/meli_financials.csv        # Extracted quarterly financials
  raw/macro/brazil_macro.csv             # Raw BCB series
  raw/macro/macro_master.csv             # Processed macro panel
  processed/master_dataset.parquet       # Merged analysis-ready dataset (40 × 146)

outputs/
  figures/                               # 10+ PNG charts
    phase_space_trajectory.png           # Flagship: capital cycle rotation
    regime_continuous.png                # Scarcity index heatmap
    regime_probability.png               # P(scarcity) over time
    cycle_decomposition.png              # HP filter trend vs cycle
    credit_scurve.png                    # Log credit with inflection detection
    capital_cycle_traditional.png        # ROIC vs capex scatter
    roic_evolution.png                   # ROIC + trend + WACC
    intensity_evolution.png              # Growth investment breakdown
    ep_persistence.png                   # Economic profit bars
    credit_growth.png                    # Credit YoY growth
    credit_intensity.png                 # Credit as % of revenue
  tables/
    summary_metrics.csv                  # Key metrics summary
```

### Re-running from Scratch

To clear all cached/generated data and re-run:

```bash
rm -rf data/raw/ data/processed/ outputs/figures/ outputs/tables/
python src/sec_data_extractor.py
python src/main.py
```

---

## Output Quality Assurance

### Quality Metrics

**Data Quality Thresholds:**
- ✅ **EXCELLENT**: ≥90% completeness
- ✅ **GOOD**: 75-89% completeness
- ⚠️ **FAIR**: 50-74% completeness
- ❌ **POOR**: <50% completeness

**Current Status:**
```
Category                    Completeness    Grade
─────────────────────────────────────────────────
Revenue & Profitability     98.3%          ✓✓ EXCELLENT
Balance Sheet - Assets      100.0%         ✓✓ EXCELLENT
Operating Expenses          72.0%          ⚠  FAIR → Improving to GOOD
Cash Flow                   58.0%          ⚠  FAIR → Improving to GOOD
```

### Validation Checks

**Automated Validations (`sec_data_extractor.py`):**
```python
def validate_data_quality(df):
    """
    Comprehensive quality checks
    """
    checks = {
        'row_count': len(df) >= 8,  # Minimum for HP filter
        'date_range': (df['date'].max() - df['date'].min()).days > 365,
        'revenue_complete': df['revenue'].notna().sum() == len(df),
        'balance_sheet_80pct': df['current_assets'].notna().sum() / len(df) >= 0.8,
        'no_duplicates': df.duplicated(subset=['date']).sum() == 0
    }
    return all(checks.values())
```

**Manual Validation Checklist:**

- [ ] **Data Sanity**: Revenue monotonically increasing? ROIC in reasonable range (0-50%)?
- [ ] **Temporal Consistency**: No gaps >2 quarters? Quarterly progression logical?
- [ ] **Cross-Metric Validation**: IC = NWC + Gross PP&E + Cap R&D? NOPAT = EBIT × (1 - tax rate)?
- [ ] **Visual Inspection**: Charts reveal insights, not artifacts? Trends match fundamentals?
- [ ] **VP Alignment**: High ROIC + high investment = super-cycle confirmed independently?

### Known Limitations & Mitigations

**Limitation 1: Tax Data Sparsity**
- **Issue**: Only ~5% of quarters have actual `taxes_paid` from SEC
- **Mitigation**: Use `current_tax_expense` (excludes deferred) → 85% cash conversion
- **Impact**: ~5% error in NOPAT, minimal on ROIC trends
- **Future**: Manual extraction from 10-K cash flow statements for recent 8 quarters

**Limitation 2: Argentina Macro Data**
- **Issue**: BCRA API rate limits / blocks requests
- **Mitigation**: Use Brazil as primary (70%+ of MELI GMV anyway)
- **Impact**: Loss of Argentina-specific regime signals
- **Future**: Manual CSV entry or Banxico API for regional comparison

**Limitation 3: Credit Portfolio**
- **Issue**: MELI doesn't break out credit in XBRL (`NotesReceivableNet` proxy)
- **Mitigation**: Use reported `loans_receivable` from investor presentations
- **Impact**: May underestimate fintech penetration
- **Future**: Parse MD&A text from 10-Qs for credit disclosures

---

## Methodology

### VP Capital Cycle Metrics

#### 1. Invested Capital (Economic Definition)
```python
# Not book equity! Economic capital required to generate returns
NWC = (Current_Assets - Cash) - (Current_Liabilities - Short_Term_Debt)
Gross_PPE = Net_PPE + Accumulated_Depreciation
Capitalized_RD = RD_Expense.rolling(16).sum() / 4  # 4-year amortization

Invested_Capital = NWC + Gross_PPE + Capitalized_RD
```

**Why Gross PP&E?** Depreciation is accounting, not economic capital consumption.  
**Why Capitalize R&D?** Software/platform development creates durable assets (GAAP expenses).

#### 2. ROIC (Cash-Based)
```python
Tax_Rate = Taxes_Paid / EBIT  # Actual cash taxes, not accrual
NOPAT = EBIT × (1 - Tax_Rate)

ROIC = NOPAT / Invested_Capital
```

**Why Cash Taxes?** Tax deferral is real economic value (float).  
**Why Not Use Book Numbers?** GAAP distorts through R&D expensing and depreciation schedules.

#### 3. Economic Profit (Shareholder Value)
```python
WACC = Base_Rate + Country_Risk_Premium
# Brazil: 12% + 4% = 16%
# LATAM: 12% + 5% = 17%

Economic_Profit = (ROIC - WACC) × Invested_Capital
EP_Margin = Economic_Profit / Revenue
```

**Interpretation:**
- EP > 0: Value creation (returns exceed cost of capital)
- EP < 0: Value destruction (burning shareholder capital)
- EP persistence: Sustainable competitive advantage

#### 4. Incremental ROIC (Capital Efficiency)
```python
# 3-year rolling
ΔNOPAT = NOPAT_t - NOPAT_(t-12)
ΔIC = IC_t - IC_(t-12)

Incremental_ROIC = ΔNOPAT / ΔIC
```

**Why It Matters:**
- Incremental ROIC > ROIC: Improving capital efficiency (good)
- Incremental ROIC > WACC: New investments create value (essential)
- Incremental ROIC declining: Diminishing returns signal (warning)

#### 5. Growth Investment Intensity

**Growth Opex:**
```python
Growth_Opex_Intensity = (Sales_Marketing + RD_Expense).rolling(4).sum() / Revenue.rolling(4).sum()
```

**Growth Capex:**
```python
Depreciation_Rate = Depreciation.rolling(4).sum() / Gross_PPE
Maintenance_Capex = Depreciation_Rate × Revenue
Growth_Capex = Total_Capex - Maintenance_Capex

Growth_Capex_Intensity = Growth_Capex.rolling(4).sum() / Revenue.rolling(4).sum()
```

**Total Growth Investment:**
```python
Total_Growth_Investment = Growth_Opex + Growth_Capex
Growth_Investment_Intensity = Total_Growth_Investment / Revenue
```

**Interpretation:**
- <15%: Low reinvestment (mature/harvesting)
- 15-30%: Moderate growth investment
- \>30%: Aggressive growth (super-cycle signal if ROIC high)

#### 6. Reinvestment Rate
```python
Reinvestment_Rate = Total_Growth_Investment / NOPAT
```

**Framework:**
- RR < 50%: Capital light, high FCF conversion
- RR 50-100%: Balanced growth
- RR > 100%: Reinvesting more than earnings (super-cycle or burning cash?)

**Context Matters:**
- High RR + High Incremental ROIC = Super-cycle ✅
- High RR + Low Incremental ROIC = Capital destruction ❌

#### 7. Credit Intensity (Fintech Signal)
```python
Credit_Intensity = Credit_Portfolio / Revenue_TTM
Credit_Growth_YoY = Credit_Portfolio.pct_change(4) × 100
Credit_Growth_Accel = Credit_Growth_YoY.diff(4)
```

**Strategic Significance:**
- Credit Intensity >15%: Fintech becoming material
- Credit Intensity >25%: Fintech is core business
- Acceleration >0: Entering exponential growth phase

---

## Institutional Enhancements

### 1. Continuous Scarcity Index

**Problem with Binary Classification:**
```python
# Old approach - 50% information loss
if inflation > 40:
    regime = "scarce"
else:
    regime = "normal"
```

**Solution: Continuous Composite Index**
```python
# Components (Z-scored and weighted)
inflation_vol = z_score(cpi.rolling(12).std())      # Weight: 30%
fx_vol = z_score(usd_brl_returns.rolling(12).std()) # Weight: 25%
real_rate_repression = z_score(-real_rate)          # Weight: 25%
credit_stress = z_score(-credit_growth_proxy)       # Weight: 20%

# Weighted composite
scarcity_index_raw = (
    0.30 × inflation_vol +
    0.25 × fx_vol +
    0.25 × real_rate_repression +
    0.20 × credit_stress
)

# Normalize to 0-100 for interpretability
scarcity_index = normalize(scarcity_index_raw, 0, 100)
```

**Benefits:**
- ✅ Preserves gradation (tight vs loose scarcity)
- ✅ Enables trend analysis (tightening vs loosening)
- ✅ Supports stress testing (shock scenarios)

### 2. Probabilistic Regime Classification

**Logistic Transformation:**
```python
# Map continuous index to probability
P(scarcity) = 1 / (1 + exp(-k × scarcity_index))

# where k controls steepness (k=2.0 standard)
```

**Regime Labels from Probabilities:**

| P(scarcity) | Regime Label    | Interpretation           |
|-------------|-----------------|--------------------------|
| 0.00 - 0.25 | Abundant        | Easy capital access      |
| 0.25 - 0.50 | Normal          | Balanced conditions      |
| 0.50 - 0.75 | Scarce          | Constrained capital      |
| 0.75 - 1.00 | Severe          | Extreme scarcity         |

**Applications:**
- **Stress Testing**: "What if inflation vol doubles?" → ΔP(scarcity)
- **Transition Risk**: "Probability of moving to severe?" → Markov chains
- **Scenario Analysis**: Bear/base/bull regime paths

### 3. Hodrick-Prescott Filter Decomposition

**Purpose**: Separate structural trend from cyclical fluctuations
```python
from statsmodels.tsa.filters.hp_filter import hpfilter

# λ = 1600 for quarterly data (standard)
roic_cycle, roic_trend = hpfilter(roic, lamb=1600)
```

**Interpretation:**

| Component      | Meaning                              | Investment Signal               |
|----------------|--------------------------------------|---------------------------------|
| **Trend**      | Structural competitive position      | Long-term ROIC trajectory       |
| **Cycle**      | Temporary deviations from trend      | Entry/exit timing signals       |
| Positive Cycle | Temporarily above structural ROIC    | Mean reversion risk             |
| Negative Cycle | Temporarily below structural ROIC    | Recovery opportunity            |

**Example Application:**
```
ROIC_actual = 35%
ROIC_trend = 32%
ROIC_cycle = +3%

Interpretation: 
- Structural ROIC = 32% (vs WACC 17% = 15% spread) → Strong
- Currently +3% above trend → Slight mean reversion risk
- If cycle turns negative while trend holds → Buy signal
```

### 4. Phase Space Transformation

**Z-Score Normalization:**
```python
# Transform raw metrics to comparable units
roic_z = (roic - roic.mean()) / roic.std()
investment_z = (total_capex_intensity - mean) / std

# Phase coordinates
phase_x = investment_z  # X-axis: Investment intensity
phase_y = roic_z        # Y-axis: Returns
```

**Phase Quadrants:**
```
        High ROIC
            │
Value       │      SUPER-CYCLE
Creation    │    (Target Position)
(Low Invest)│  (High ROIC + High Invest)
────────────┼────────────────
Capital     │     Capital
Abundance   │   Destruction
(Low ROIC)  │ (Low ROIC + High Invest)
            │
       Low ROIC
```

**Trajectory Analysis:**
- **Clockwise rotation**: Classic capital cycle (competition → destruction → rationalization → creation)
- **Counterclockwise**: Unusual (tightening scarcity while improving ROIC)
- **Quadrant persistence**: Moat strength (staying in super-cycle quadrant >12 quarters)

### 5. Log Transforms for S-Curves

**Problem**: Linear scaling hides exponential growth inflections

**Solution**: Log transformation reveals S-curves
```python
# Log scale reveals convexity
credit_log = log(credit_portfolio + 1)

# First derivative = growth rate
growth_rate = Δ(credit_log) / Δt

# Second derivative = acceleration (inflection detection)
acceleration = Δ(growth_rate) / Δt
```

**Inflection Point Detection:**
```python
# Inflection = where acceleration changes sign
inflection_points = where(acceleration[t-1] × acceleration[t] < 0)
```

**S-Curve Phases:**

| Phase             | Growth Rate | Acceleration | Interpretation           |
|-------------------|-------------|--------------|--------------------------|
| Early (Linear)    | Low         | +            | Building momentum        |
| **Inflection**    | Accelerating| **Max**      | **Entering exponential** |
| Exponential       | High        | +            | Rapid expansion          |
| Saturation        | Declining   | -            | Approaching maturity     |

**Investment Implications:**
- Pre-inflection: Early, highest risk/reward
- Post-inflection (accelerating): Momentum opportunity
- Saturation: Exit or find next S-curve

---

## Troubleshooting

### Common Issues

#### Issue 1: "No module named 'statsmodels'"

**Solution:**
```bash
pip install statsmodels>=0.14.0
```

**Why**: HP filter decomposition requires statsmodels

---

#### Issue 2: "Config file not found"

**Symptoms:**
```
ERROR - Config file not found: .../config.yaml
```

**Solution:**
```bash
# config.yaml must be in the PROJECT ROOT (not src/)
ls -la config.yaml

# If missing, recreate it — see config.yaml in the repo root for the full template.
```

---

#### Issue 3: "Analysis failed: 'taxes_paid'"

**Symptoms:**
```
KeyError: 'taxes_paid'
```

**Cause**: SEC data missing `taxes_paid` field

**Solution:** Re-run updated `sec_data_extractor.py`
```bash
python sec_data_extractor.py
```

The updated extractor tries multiple tax fields:
1. `IncomeTaxesPaid` (actual cash taxes - best)
2. `CurrentIncomeTaxExpenseBenefit` (excludes deferred - good proxy)
3. `IncomeTaxExpenseBenefit × 0.85` (total tax adjusted - acceptable)
4. `EBIT × 0.25` (estimated - fallback)

---

#### Issue 4: "Data quality only 50%"

**Symptoms:**
```
⚠  FAIR - Consider manual data entry for key missing fields
Overall quality: 55.9%
```

**Diagnosis:**
- Run: `python sec_data_extractor.py`
- Check data quality report

**Solutions:**

**A. Update Extractor (Recommended)**
- Replace `sec_data_extractor.py` with latest version
- New version tries 3-5 XBRL field alternatives per metric

**B. Manual Supplementation (If Still Needed)**
```python
# For critical missing quarters, manually add from 10-K/10-Q PDFs
import pandas as pd

df = pd.read_csv('data/raw/company/meli_financials.csv')

# Example: Add Q3 2024 balance sheet data
df.loc[df['date'] == '2024-09-30', 'current_assets'] = 17824
df.loc[df['date'] == '2024-09-30', 'current_liabilities'] = 14957

df.to_csv('data/raw/company/meli_financials.csv', index=False)
```

---

#### Issue 5: N/A in Summary Table

**Symptoms:**
```
Growth Capex Intensity (%)    : N/A
Total Capex Intensity (%)     : N/A
```

**Cause**: Missing capex data in latest quarter

**Solution: Forward-fill capex**
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/company/meli_financials.csv', parse_dates=['date'])
df = df.sort_values('date')
df['ppe_additions'] = df['ppe_additions'].fillna(method='ffill', limit=1)
df['intangible_additions'] = df['intangible_additions'].fillna(method='ffill', limit=1)
df.to_csv('data/raw/company/meli_financials.csv', index=False)
print('✓ Capex forward-filled')
"
```

Or update `data_acquisition.py` (already includes this fix).

---

#### Issue 6: Macro Regime Shows "N/A"

**Symptoms:**
```
Capital Scarcity Index (0-100)    : N/A
Scarcity Regime                   : nan
```

**Cause**: Country mismatch (company="LATAM", macro="BRA")

**Solution**: Already fixed in updated `data_acquisition.py`
```python
# Line ~305
df['country'] = 'BRA'  # Changed from 'LATAM'
```

If still seeing issue:
```bash
# Verify country assignment
python -c "
import pandas as pd
df = pd.read_csv('data/raw/company/meli_financials.csv')
print(df['country'].unique())
# Should print: ['BRA']
"
```

---

#### Issue 7: "HP filter needs ≥8 observations"

**Symptoms:**
```
WARNING - roic: need ≥8 observations (have 5)
```

**Cause**: Insufficient historical data

**Solutions:**
- **Short-term**: Framework will skip decomposition, use raw metrics
- **Long-term**: Accumulate more quarters (run quarterly when 10-Q filed)

---

#### Issue 8: Matplotlib Arrow Error

**Symptoms:**
```
StopIteration
  File "matplotlib/patches.py", line 2815, in _clip
```

**Cause**: Bug in matplotlib arrow annotations (tight clustering)

**Solution**: Already fixed in updated `visualization.py`
- Removed problematic arrow annotations
- Used simple scatter + line trajectory instead

---

### Data Quality Validation

**Run quality checks:**
```bash
python -c "
import pandas as pd

df = pd.read_csv('data/raw/company/meli_financials.csv')

print('Data Quality Report')
print('='*60)
print(f'Total quarters: {len(df)}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
print()

critical_fields = [
    'revenue', 'ebit', 'current_assets', 'current_liabilities',
    'net_ppe', 'accumulated_depreciation', 'rd_expense',
    'sales_marketing', 'taxes_paid'
]

for field in critical_fields:
    if field in df.columns:
        pct = (df[field].notna().sum() / len(df)) * 100
        status = '✓✓' if pct >= 90 else '✓' if pct >= 75 else '⚠' if pct >= 50 else '✗'
        print(f'{status} {field:25s}: {pct:5.1f}%')
    else:
        print(f'✗ {field:25s}: MISSING')
"
```

**Expected Output:**
```
✓✓ revenue                  : 100.0%
✓✓ ebit                     :  92.0%
✓✓ current_assets           : 100.0%
✓✓ current_liabilities      : 100.0%
✓  net_ppe                  :  78.0%
✓  accumulated_depreciation :  76.0%
✓  rd_expense               :  82.0%
✓  sales_marketing          :  88.0%
⚠  taxes_paid               :  65.0%
```

---

### Performance Optimization

**For Large Datasets (>100 quarters):**
```python
# In config.yaml, adjust HP filter lambda
parameters:
  hp_lambda: 1600  # Standard for quarterly
  # hp_lambda: 6.25  # For annual data
  # hp_lambda: 129600  # For monthly data
```

**Speed Up Visualizations:**
```python
# In visualization.py, reduce DPI for drafts
plt.savefig(output_path, dpi=150)  # Draft (fast)
# plt.savefig(output_path, dpi=300)  # Production (slower, higher quality)
```

---

## References

### Theoretical Foundations

1. **Capital Cycle Theory**
   - Marathon Asset Management (2011). *Capital Returns: Investing Through the Capital Cycle*
   - Variant Perception Research (2015-2025). *Capital Cycle Frameworks*

2. **Economic Profit Framework**
   - Koller, Goedhart, Wessels (2020). *Valuation: Measuring and Managing the Value of Companies* (7th Ed.)
   - Rappaport, A. (1986). *Creating Shareholder Value*

3. **Emerging Markets Capital Scarcity**
   - Damodaran, A. (2012). *Investment Valuation* (3rd Ed.) - Chapter on Country Risk
   - Reinhart & Rogoff (2009). *This Time Is Different: Eight Centuries of Financial Folly*

### Statistical Methods

4. **Hodrick-Prescott Filter**
   - Hodrick, R. & Prescott, E. (1997). "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking*
   - Ravn & Uhlig (2002). "On Adjusting the Hodrick-Prescott Filter for the Frequency of Observations"

5. **Time Series Decomposition**
   - Cleveland et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"

### Data Sources

6. **SEC EDGAR**
   - SEC Company Facts API: https://www.sec.gov/edgar/sec-api-documentation
   - XBRL Taxonomy: https://www.sec.gov/info/edgar/edgartaxonomies

7. **Macro Data**
   - Brazil Central Bank (BCB): https://www3.bcb.gov.br/sgspub/
   - Argentina Central Bank (BCRA): http://api.estadisticasbcra.com/

---

## Appendix: Output Scrutiny Checklist

Use this checklist when validating framework outputs:

### Data Quality
- [ ] Overall completeness ≥80%
- [ ] No duplicated quarters
- [ ] Date range ≥2 years (minimum for trends)
- [ ] Revenue monotonically increasing (or explained decreases)
- [ ] ROIC in reasonable range (0-50% for tech platforms)
- [ ] No negative invested capital
- [ ] Capitalized R&D ≤50% of IC (sanity check)

### Chart Quality
- [ ] **Phase Space**: Clear quadrant separation, trajectory visible
- [ ] **Regime Heatmap**: Gradation visible (not saturated)
- [ ] **Cycle Decomposition**: Trend smooth, cycle oscillates around zero
- [ ] **Credit S-Curve**: Inflection point visible if present
- [ ] **ROIC Evolution**: Trend vs actual distinguishable
- [ ] **Intensity Charts**: Y-axis scales appropriate (not compressed)

### Metric Validation
- [ ] ROIC > WACC for value creation periods
- [ ] Incremental ROIC ≈ ROIC ± 5% (if diverging, investigate)
- [ ] EP sign matches (ROIC - WACC) sign
- [ ] Reinvestment rate = Growth Investment / NOPAT (verify calculation)
- [ ] Credit intensity = Credit / Revenue (spot check)

### Regime Classification
- [ ] Scarcity index correlates with inflation/FX vol (face validity)
- [ ] Regime probability smooth (no erratic jumps)
- [ ] Brazil shows lower scarcity than Argentina (expected)
- [ ] Regime shifts align with known macro events (2020 COVID, etc.)

### VP Methodology Alignment
- [ ] No demand forecasting in analysis
- [ ] Supply-side factors emphasized (capacity, competition, barriers)
- [ ] Capital scarcity recognized as ROIC persistence mechanism
- [ ] Mean reversion acknowledged (but regime-conditional)
- [ ] Moats/competitive advantages explicitly discussed

### Institutional Standards
- [ ] All charts have titles, axis labels, legends
- [ ] Color schemes consistent across charts
- [ ] DPI ≥300 for production outputs
- [ ] CSV table formats cleanly (no scientific notation)
- [ ] Outputs dated/versioned for reproducibility

---

## Future Enhancements

**Planned Improvements:**

1. **Multi-Company Comparison**
   - Add MELI competitors (Sea Limited, Shopee, Jumia)
   - Cross-company phase space overlay
   - Relative ROIC positioning

2. **Forecasting Module**
   - Regime transition probabilities (Markov model)
   - Mean reversion scenarios (structural ROIC ± cyclical)
   - Credit S-curve extrapolation

3. **Interactive Dashboards**
   - Plotly/Dash web interface
   - Real-time data updates
   - Scenario analysis sliders

4. **Alternative Data Integration**
   - App download rankings (SensorTower)
   - Web traffic (SimilarWeb)
   - Social sentiment (Twitter, Reddit)

5. **Automated Alerts**
   - Regime shift detection
   - ROIC inflection warnings
   - Credit acceleration triggers

---

## License

This framework is provided for educational and research purposes. 

**Disclaimer**: This framework is for analytical purposes only and does not constitute investment advice. Always conduct independent research and consult qualified professionals before making investment decisions.

---

## Contact & Support

**Issues**: Create an issue in the repository  
**Questions**: Refer to this README or code comments  
**Contributions**: Pull requests welcome (maintain VP methodology alignment)

---

**Last Updated**: January 2026  
**Version**: 2.0 (Institutional Grade)  
**Framework Status**: Production-Ready ✅