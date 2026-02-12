# VP Capital Cycle Analysis - MercadoLibre

Institutional-grade capital cycle analysis framework implementing VP/HOLT methodology for MercadoLibre Inc. (MELI). This system provides sophisticated financial metrics, regime modeling, and visualization tools for analyzing capital allocation efficiency and economic profit persistence.

## Overview

This project performs comprehensive capital cycle analysis using:

- **VP/HOLT Adjusted ROIC**: Capitalizes R&D, S&M, and software investments to calculate economic returns
- **Capital Scarcity Regime Analysis**: Continuous probabilistic regime classification using macro indicators
- **Cycle Decomposition**: HP-filter trend/cycle separation for ROIC and investment intensity
- **Phase Space Analysis**: Four-quadrant capital cycle positioning
- **Credit S-Curve Detection**: Identifies inflection points in credit monetization
- **Economic Profit Persistence**: Tracks value creation dynamics over time

## Features

### Financial Metrics
- Traditional & Adjusted ROIC (VP/HOLT methodology)
- Incremental ROIC (3-year rolling)
- Economic Profit (EP) and EP Margin
- ROIC vs WACC spread analysis
- Capital intensity decomposition (growth vs maintenance)

### Advanced Analytics
- **Regime Modeling**: Continuous capital scarcity index (0-100 scale)
- **Probabilistic Classification**: Logistic regime probabilities
- **Cycle Decomposition**: Structural trends vs cyclical deviations
- **Phase Space Trajectory**: Investment intensity vs returns mapping
- **Credit Growth Analysis**: Log-scale S-curve and acceleration metrics

### Visualizations
- Phase space trajectory diagrams
- Regime probability heatmaps
- ROIC evolution (reported vs adjusted)
- Capital intensity trends
- Economic profit persistence charts
- Credit S-curve and growth dynamics

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
# SSH (recommended if you have GitHub SSH keys)
git clone git@github.com:HelloValentino/cyclical-analysis-meli.git
cd cyclical-analysis-meli

# Or HTTPS
git clone https://github.com/HelloValentino/cyclical-analysis-meli.git
cd cyclical-analysis-meli
```

2. **Create and activate virtual environment**
```bash
python3 -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install pandas numpy requests scipy statsmodels matplotlib seaborn pyyaml
```

Or create a `requirements.txt`:
```txt
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
scipy>=1.11.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

Then install:
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import pandas, numpy, scipy, statsmodels; print('All dependencies installed successfully')"
```

## Configuration

### 1. Configuration File

Edit [config.yaml](config.yaml) to customize analysis parameters:

```yaml
paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  outputs_figures: "outputs/figures"
  outputs_tables: "outputs/tables"

companies:
  primary:
    ticker: "MELI"
    name: "MercadoLibre Inc."

time_periods:
  start_date: "2015-01-01"  # Analysis start
  end_date: "2025-12-31"    # Analysis end

parameters:
  base_wacc: 0.12                    # Base WACC (12%)
  country_risk_premium:
    ARG: 0.08                        # Argentina risk premium
    BRA: 0.04                        # Brazil risk premium
    MEX: 0.02                        # Mexico risk premium
    LATAM: 0.05                      # LATAM average
  rd_amortization_years: 4.0         # R&D capitalization period
  regime_thresholds:
    inflation_high: 40.0             # High inflation threshold (%)
    inflation_moderate: 15.0
    fx_vol_high: 0.15
    real_rate_negative: 0
    credit_gap_negative: -5.0
```

### 2. Data Requirements

The analysis requires financial data in CSV format:

**File**: `data/raw/company/meli_financials.csv`

**Required columns**:
- `date` (YYYY-MM-DD format)
- `ticker` (e.g., "MELI")
- `period_type` ("10-Q" or "10-K")
- `revenue`, `cogs`, `ebit`, `rd_expense`, `sales_marketing`
- `current_assets`, `current_liabilities`, `cash`
- `net_ppe`, `accumulated_depreciation`
- `short_term_debt`, `long_term_debt`
- `taxes_paid`, `tax_expense`
- `ppe_additions`, `intangible_additions`
- `credit_portfolio` (optional)
- `software_investment` (optional)

**To extract SEC data** (if you have the extractor):
```bash
python src/sec_data_extractor.py
```

## Usage

### View Outputs with Dashboard (Recommended)

The project includes an interactive HTML dashboard for browsing all outputs, visualizations, and documentation:

```bash
# From project root, start a local web server
python -m http.server 8000

# Then open in your browser:
# http://localhost:8000/dashboard.html
```

**Dashboard Features:**
- **Visualizations**: View all 11 charts (phase space, regime analysis, ROIC evolution, etc.)
- **Data Tables**: Download summary metrics CSV and master dataset (Parquet)
- **Documentation**: Rendered README and data quality docs via built-in Markdown viewer
- **Quick Start**: Code snippets for running the pipeline

**Note**: The dashboard requires a local server because browsers block cross-file requests for `file://` URLs. The simple Python HTTP server is the easiest solution.

### Run the Pipeline

**Option 1: Automated Script (Recommended)**

The easiest way to run the pipeline is using the included script that clears cache and runs the analysis:

```bash
./run_pipeline.sh
```

This script will:
- Clear all cache directories (macro data, processed data, outputs)
- Verify virtual environment exists
- Run the complete analysis pipeline
- Show a summary of generated outputs

**Option 2: Manual Execution**

Run the pipeline directly without clearing cache:

```bash
python src/main.py

# Or using the virtual environment explicitly:
.venv/bin/python src/main.py
```

### Expected Output

The analysis will:

1. **Fetch Macro Data** - Brazil economic indicators from BCB (Central Bank)
   - CPI (inflation)
   - SELIC (policy rate)
   - USD/BRL exchange rate
   - Calculated volatility metrics

2. **Load Company Financials** - From CSV file

3. **Calculate VP Metrics** - ROIC, NOPAT, invested capital, etc.

4. **Apply Advanced Analytics** - Regime analysis, cycle decomposition

5. **Merge Datasets** - Company + macro data

6. **Generate Visualizations** - 10+ institutional-grade charts

7. **Create Summary Tables** - Key metrics CSV

### Output Structure

```
outputs/
├── figures/
│   ├── phase_space_trajectory.png        # Capital cycle positioning
│   ├── capital_cycle_traditional.png     # ROIC vs Capex intensity
│   ├── regime_continuous.png             # Scarcity regime heatmap
│   ├── regime_probability.png            # Regime probabilities over time
│   ├── cycle_decomposition.png           # Trend/cycle separation
│   ├── roic_evolution.png                # ROIC: reported vs adjusted
│   ├── intensity_evolution.png           # Investment intensity trends
│   ├── ep_persistence.png                # Economic profit dynamics
│   ├── credit_scurve.png                 # Credit S-curve analysis
│   ├── credit_growth.png                 # Credit growth dynamics
│   └── credit_intensity.png              # Credit deployment intensity
└── tables/
    └── summary_metrics.csv               # Latest metrics summary

data/
└── processed/
    └── master_dataset.parquet            # Full merged dataset
```

### Example Output

```
================================================================================
VP CAPITAL CYCLE ANALYSIS - MERCADOLIBRE
INSTITUTIONAL-GRADE FRAMEWORK WITH VP/HOLT ADJUSTED ROIC
================================================================================

STEP 1: FETCHING MACRO DATA
✓ Fetched cpi: 131 observations
✓ Fetched selic: 2847 observations (2 chunk(s))
✓ Fetched usd_brl: 2845 observations (2 chunk(s))

STEP 2: LOADING COMPANY FINANCIALS
✓ Loaded 45 periods
  Date range: 2015-03-31 to 2024-12-31
  Ticker: ['MELI']

STEP 3: CALCULATING VP METRICS (WITH ADJUSTED ROIC)
✓ Invested Capital calculated
✓ Return metrics calculated
✓ Capital intensity metrics calculated
✓ VP/HOLT Adjusted ROIC calculated

STEP 4: APPLYING INSTITUTIONAL ANALYTICS
✓ Scarcity index range: 12.3 - 87.6
✓ Cycle decomposition complete

STEP 5: MERGING DATASETS
✓ Merged dataset created
  Rows: 45 | Cols: 127

STEP 6: GENERATING INSTITUTIONAL SUMMARY
STEP 7: GENERATING VISUALIZATIONS
✓ ALL VISUALIZATIONS COMPLETE

STEP 8: GENERATING SUMMARY TABLE

================================================================================
INSTITUTIONAL INSIGHTS
================================================================================
Capital Cycle Phase: super_cycle
Scarcity Regime: scarce
Scarcity Probability: 68.4%
Super-Cycle Signal: CONFIRMED ✓

Adjusted ROIC: 23.5%  |  WACC: 16.0%  |  Spread: 750pp

✓ ANALYSIS COMPLETE
```

## Project Structure

```
cyclical-analysis-meli/
├── README.md                    # This file
├── config.yaml                  # Configuration parameters
├── run_pipeline.sh              # Pipeline runner script (clears cache + runs analysis)
├── dashboard.html               # Interactive output browser (recommended)
├── md-viewer.html               # Markdown documentation viewer
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   ├── main.py                  # Main orchestration script
│   ├── data_acquisition.py      # Macro data fetching (BCB API)
│   ├── calculations.py          # VP/HOLT metrics calculator
│   ├── analysis.py              # Dataset merging logic
│   ├── advanced_analysis.py     # Regime modeling & cycle decomposition
│   ├── visualization.py         # Chart generation
│   └── sec_data_extractor.py    # SEC data extraction (optional)
│
├── data/                        # Data directory
│   ├── raw/
│   │   ├── company/             # Company financials
│   │   │   └── meli_financials.csv
│   │   ├── macro/               # Macro data (auto-generated)
│   │   │   ├── brazil_macro.csv
│   │   │   └── macro_master.csv
│   │   └── sec_cache/           # SEC API cache (optional)
│   └── processed/
│       └── master_dataset.parquet  # Final merged dataset
│
├── outputs/                     # Analysis outputs
│   ├── figures/                 # PNG visualizations
│   └── tables/                  # CSV summary tables
│
├── docs/                        # Documentation
│   ├── data_quality.md          # Data quality & preparation guide
│   └── definitions.md           # Comprehensive metrics glossary
│
└── .venv/                       # Virtual environment (not in git)
```

## Key Concepts

### VP/HOLT Adjusted ROIC

Traditional ROIC treats R&D and S&M as expenses. VP/HOLT methodology capitalizes these investments:

**Adjustments**:
- **R&D**: Capitalized over 4 years (configurable)
- **Sales & Marketing**: Capitalized over 2 years (8 quarters)
- **Software**: Capitalized over 4 years (16 quarters)
- **Credit Assets**: Adjusted for excess cash offset

**Formula**:
```
Adjusted ROIC = Adjusted NOPAT (TTM) / Adjusted Invested Capital (Avg)

Where:
  Adjusted NOPAT = EBIT × (1 - Cash Tax Rate) + Capitalizations - Amortizations
  Adjusted IC = Gross PPE + Cap R&D + Cap S&M + Cap Software + NWC + Operating Credit
```

### Capital Scarcity Regime

Continuous index (0-100) measuring capital market conditions:

**Components**:
- Inflation volatility (30% weight)
- FX volatility (25% weight)
- Real rate repression (25% weight)
- Credit collapse proxy (20% weight)

**Regime Classification**:
- **Abundant**: P(Scarcity) < 25%
- **Normal**: 25% ≤ P(Scarcity) < 50%
- **Scarce**: 50% ≤ P(Scarcity) < 75%
- **Severe**: P(Scarcity) ≥ 75%

### Phase Space Quadrants

Four-quadrant framework mapping investment vs returns:

1. **Super Cycle** (High ROIC, High Investment): Attractive growth
2. **Value Creation** (High ROIC, Low Investment): Capital discipline
3. **Capital Abundance** (Low ROIC, Low Investment): Maturity/retrenchment
4. **Capital Destruction** (Low ROIC, High Investment): Value trap

## Troubleshooting

### Common Issues

**1. "Config file not found"**
```bash
# Ensure you're in project root:
pwd  # Should show: .../cyclical-analysis-meli

# Check config exists:
ls config.yaml
```

**2. "Financial data not found"**
```bash
# Create required directory:
mkdir -p data/raw/company

# Verify file exists:
ls data/raw/company/meli_financials.csv

# If missing, extract from SEC or obtain from data provider
```

**3. "No macro data available"**
```bash
# The macro fetcher may fail due to:
# - Network issues
# - BCB API rate limits
# - Date range too large

# Analysis can continue without macro data (uses neutral regime defaults)
```

**4. "statsmodels not installed"**
```bash
pip install statsmodels

# If still fails, try:
pip install --upgrade statsmodels scipy
```

**5. "Module not found" errors**
```bash
# Ensure you're in project root when running:
python src/main.py

# NOT:
cd src && python main.py  # ❌ Wrong
```

## Data Sources

### Macro Data (Automatic)
- **Brazil Central Bank (BCB)**: CPI, SELIC, USD/BRL via SGS API
- Fetched automatically during analysis
- Cached in `data/raw/macro/`

### Company Data (Manual)
- **SEC EDGAR**: 10-Q and 10-K filings
- Requires extraction or manual CSV preparation
- Place in `data/raw/company/meli_financials.csv`

## Advanced Usage

### Custom Analysis Period

```yaml
# In config.yaml:
time_periods:
  start_date: "2018-01-01"  # Change as needed
  end_date: "2024-12-31"
```

### Adjust WACC Components

```yaml
parameters:
  base_wacc: 0.10           # Lower base WACC to 10%
  country_risk_premium:
    BRA: 0.03               # Reduce Brazil risk premium
```

### Change R&D Capitalization

```yaml
parameters:
  rd_amortization_years: 3.0  # Shorter useful life
```

### Programmatic Access

```python
import pandas as pd

# Load processed dataset
df = pd.read_parquet("data/processed/master_dataset.parquet")

# Access key metrics
latest = df.iloc[-1]
print(f"Adjusted ROIC: {latest['adjusted_roic']:.2%}")
print(f"ROIC Spread: {(latest['adjusted_roic'] - latest['wacc']) * 100:.1f}pp")
print(f"Phase: {latest['phase_quadrant']}")
```

## Performance Notes

- **Macro fetch**: ~10-30 seconds (depends on BCB API response)
- **Calculations**: <5 seconds for typical dataset (40-50 periods)
- **Visualizations**: ~5-10 seconds (10+ charts)
- **Total runtime**: ~30-60 seconds

## Contributing

### Development Setup

```bash
# Install dev dependencies
pip install pytest black flake8 mypy

# Run linter
flake8 src/

# Format code
black src/

# Type checking
mypy src/
```

### Code Style

- Follow PEP 8
- Add docstrings to all public functions
- Include type hints
- Maintain defensive error handling

## License

This project is for educational and research purposes. Financial data sources may have their own usage restrictions.

## Acknowledgments

- **VP/HOLT Methodology**: Credit Suisse HOLT
- **Capital Cycle Framework**: Marathon Asset Management
- **Data Sources**: SEC EDGAR, Brazil Central Bank (BCB)

## Contact

For questions or issues, please open an issue in the repository.

---

**Last Updated**: February 2026
**Version**: 1.0.0
**Python**: 3.9+
