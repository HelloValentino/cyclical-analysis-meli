# Key Metrics & Definitions

Comprehensive glossary of financial metrics, capital cycle concepts, and analytical frameworks used in this VP Capital Cycle Analysis.

---

## Core Return Metrics

### ROIC (Return on Invested Capital)

**Definition**: Measures how efficiently a company generates returns from its invested capital.

**Formula (Traditional)**:
```
ROIC = NOPAT / Average Invested Capital
Where:
  NOPAT = Net Operating Profit After Tax
  Invested Capital = Total Assets - Non-Interest-Bearing Current Liabilities
```

**Interpretation**:
- **> WACC**: Value creation (economic profit positive)
- **= WACC**: Break-even (zero economic profit)
- **< WACC**: Value destruction (economic profit negative)

**Typical Ranges**:
- High performers: 15-25%+
- Average companies: 8-12%
- Struggling businesses: <8%

---

### Adjusted ROIC (VP/HOLT Methodology)

**Definition**: Enhanced ROIC calculation that capitalizes intangible investments (R&D, S&M, software) rather than expensing them immediately.

**Key Adjustments**:
1. **Capitalized R&D**: Amortized over 4 years (configurable)
2. **Capitalized Sales & Marketing**: Amortized over 2 years (8 quarters)
3. **Capitalized Software**: Amortized over 4 years (16 quarters)
4. **Operating Credit**: Included in invested capital

**Why It Matters**: Traditional accounting expenses R&D/S&M immediately, understating true invested capital and overstating returns for growth companies. VP/HOLT adjustment provides economic reality.

**Formula**:
```
Adjusted ROIC = Adjusted NOPAT (TTM) / Adjusted Invested Capital (Avg)

Where:
  Adjusted NOPAT = EBIT × (1 - Cash Tax Rate) + Capitalizations - Amortizations
  Adjusted IC = Gross PPE + Cap R&D + Cap S&M + Cap Software + NWC + Operating Credit
```

---

### WACC (Weighted Average Cost of Capital)

**Definition**: The average rate a company must pay to finance its assets, weighted by debt and equity proportions.

**Formula**:
```
WACC = (E/V × Re) + (D/V × Rd × (1 - Tc))

Where:
  E = Market value of equity
  D = Market value of debt
  V = E + D (total firm value)
  Re = Cost of equity (typically CAPM)
  Rd = Cost of debt
  Tc = Corporate tax rate
```

**For LATAM Companies**: Add country risk premium
```
WACC_adjusted = Base WACC + Country Risk Premium
Example for MELI: 12% base + 4% Brazil premium = 16%
```

**Interpretation**: Minimum acceptable return threshold. Projects earning below WACC destroy shareholder value.

---

### ROIC Spread

**Definition**: The difference between ROIC and WACC, measuring economic profit margin.

**Formula**:
```
Spread = Adjusted ROIC - WACC
```

**Interpretation**:
- **Positive spread**: Value creation (earning above cost of capital)
- **Zero spread**: Break-even
- **Negative spread**: Value destruction

**Example**: ROIC = 26.2%, WACC = 16.0% → Spread = +10.2pp (1,020 basis points)

---

## Capital Efficiency Metrics

### Invested Capital

**Definition (Traditional)**: Capital employed in core operations.

**Formula**:
```
Invested Capital = Total Assets - Non-Interest-Bearing Current Liabilities
OR
Invested Capital = Net Working Capital + Net PPE + Intangibles
```

**Adjusted Invested Capital (VP/HOLT)**:
```
Adjusted IC = Gross PPE + Capitalized R&D + Capitalized S&M +
              Capitalized Software + Net Working Capital + Operating Credit
```

---

### Capital Intensity

**Definition**: Capital expenditure as a percentage of revenue, measuring how much investment is required to generate sales.

**Formula**:
```
Total Capex Intensity = Total Capex / Revenue (TTM)

Where:
  Total Capex = PPE Additions + Intangible Additions
```

**Breakdown**:
- **Growth Capex**: Investment to expand capacity/revenue
- **Maintenance Capex**: Investment to sustain existing operations

**Typical Ranges**:
- Asset-light (software, services): 2-5%
- Moderate (retail, fintech): 5-10%
- Capital-intensive (manufacturing, infrastructure): 10-20%+

---

### Investment Intensity

**Definition**: Broader measure including both capex and operational growth investments.

**Formula**:
```
Total Growth Investment Intensity =
  (Growth Capex + Growth Opex Investments) / Revenue (TTM)

Where:
  Growth Opex = Capitalized R&D + Capitalized S&M + Capitalized Software
```

**Interpretation**: Higher intensity → aggressive growth mode; Lower intensity → harvest/efficiency mode

---

## Economic Profit Metrics

### Economic Profit (EP)

**Definition**: Profit earned above the cost of capital, measuring true economic value creation.

**Formula**:
```
Economic Profit = (ROIC - WACC) × Invested Capital
OR
EP = NOPAT - (WACC × Invested Capital)
```

**Interpretation**:
- **EP > 0**: Value creation
- **EP = 0**: Break-even (accounting profit but no economic profit)
- **EP < 0**: Value destruction (returns below capital cost)

**Why It Matters**: Companies can show positive net income but negative EP if ROIC < WACC.

---

### EP Margin

**Definition**: Economic profit as a percentage of revenue.

**Formula**:
```
EP Margin = Economic Profit / Revenue (TTM)
```

**Interpretation**: Higher margins indicate better quality of earnings and sustainable competitive advantage.

---

### Incremental ROIC

**Definition**: Return on new capital invested over a rolling period (typically 3 years).

**Formula**:
```
Incremental ROIC = Δ NOPAT (3yr) / Δ Invested Capital (3yr)

Where:
  Δ = Change over 3-year period
```

**Interpretation**:
- **> ROIC (TTM)**: New investments more productive than base business
- **< ROIC (TTM)**: Returns declining (diminishing returns or competitive pressure)

**Ideal**: Incremental ROIC ≥ WACC and ≥ TTM ROIC

---

## Capital Cycle Framework

### Phase Space Quadrants

Capital cycle positioning based on ROIC vs investment intensity:

#### 1. **Super Cycle** (High ROIC, High Investment)
- **Characteristics**: Attractive growth opportunities, reinvesting at high returns
- **Example**: Early-stage tech platforms, fintech expansion phase
- **Strategy**: Fund aggressively, maximize growth while returns exceed WACC
- **Risk**: Diminishing returns, competitive entry

#### 2. **Value Creation** (High ROIC, Low Investment)
- **Characteristics**: Capital discipline, high returns with modest reinvestment
- **Example**: Mature cash cows, oligopolies
- **Strategy**: Return capital to shareholders, selective growth
- **Risk**: Growth stagnation, disruption

#### 3. **Capital Abundance** (Low ROIC, Low Investment)
- **Characteristics**: Mature/declining business, retrenching
- **Example**: Legacy industrials, shrinking markets
- **Strategy**: Restructure, exit, or harvest
- **Risk**: Continued value destruction

#### 4. **Capital Destruction** (Low ROIC, High Investment)
- **Characteristics**: Value trap, pouring capital into low-return projects
- **Example**: Over-expanded retailers, failed growth stories
- **Strategy**: Cut investment, restructure, or exit
- **Risk**: Severe value destruction if continues

**Quadrant Thresholds**:
- ROIC threshold: Typically WACC or median ROIC
- Investment threshold: Median investment intensity

---

### Capital Scarcity Regime

**Definition**: Continuous index (0-100) measuring capital market conditions based on macro indicators.

**Components** (weighted):
1. **Inflation Volatility** (30%): Rolling 8-quarter std dev of CPI
2. **FX Volatility** (25%): Rolling 8-quarter std dev of exchange rate
3. **Real Rate Repression** (25%): Degree to which real rates are negative
4. **Credit Collapse Proxy** (20%): Credit growth gap vs trend

**Formula**:
```
Scarcity Index = Weighted Sum of Normalized Components × 100

P(Scarcity) = Logistic Function(Scarcity Index)
```

**Regime Classification**:
- **Abundant**: P(Scarcity) < 25% — Easy capital access, low hurdle rates
- **Normal**: 25% ≤ P(Scarcity) < 50% — Balanced conditions
- **Scarce**: 50% ≤ P(Scarcity) < 75% — Tight capital, higher hurdle rates
- **Severe**: P(Scarcity) ≥ 75% — Capital crisis, very high cost of capital

**Why It Matters**: Capital scarcity affects:
- WACC adjustments (add risk premium in scarce regimes)
- Investment hurdle rates
- Competitive dynamics (fewer new entrants in scarce capital)

---

## Advanced Analytics

### HP Filter Decomposition

**Definition**: Statistical method separating time series into structural trend and cyclical deviation.

**Components**:
1. **Trend**: Long-term structural component (slow-moving)
2. **Cycle**: Short-term deviations from trend (mean-reverting)

**Parameters**:
- Lambda (λ) = 1600 for quarterly data (standard)
- Higher λ → smoother trend, more in cycle
- Lower λ → trend tracks data more closely

**Applied To**:
- ROIC (trend = sustainable returns, cycle = temporary deviations)
- Investment intensity (trend = structural capex needs, cycle = timing)
- Economic profit (trend = moat strength, cycle = competitive pressure)

**Interpretation**:
- Large positive cycle → temporary boost (may mean-revert)
- Large negative cycle → temporary pressure (may recover)
- Trend direction → structural momentum

---

### Credit S-Curve

**Definition**: Framework for detecting inflection points in credit portfolio growth.

**Methodology**:
1. Log-transform credit portfolio (linearizes S-curve)
2. Calculate YoY growth rate
3. Calculate acceleration (2nd derivative)

**Detection**:
- **Acceleration > 0**: Early/growth phase (accelerating)
- **Acceleration ≈ 0**: Inflection point (peak growth rate)
- **Acceleration < 0**: Maturity/saturation (decelerating)

**Why It Matters**: Helps identify when credit monetization transitions from growth to maturity phase.

---

## Data Quality Metrics

### YTD Detection

**Definition**: Algorithm to identify whether reported quarterly values are standalone or year-to-date cumulative.

**Heuristic**:
```python
def _looks_ytd(q1, q2, q3):
    # Check if Q1 < Q2 < Q3 (cumulative pattern)
    if not (q1 <= q2 <= q3):
        return False
    # Check if Q2 ≈ 2×Q1 and Q3 ≈ 3×Q1 (within 30% tolerance)
    return abs(q2/q1 - 2) < 0.3 and abs(q3/q1 - 3) < 0.3
```

**Conversion** (if YTD detected):
```
Standalone Q1 = Q1_ytd
Standalone Q2 = Q2_ytd - Q1_ytd
Standalone Q3 = Q3_ytd - Q2_ytd
Standalone Q4 = FY - Q3_ytd
```

---

## Taxation Metrics

### Cash Tax Rate

**Definition**: Actual cash taxes paid as a percentage of pre-tax income.

**Formula** (with fallback hierarchy):
```
1. Cash Tax Rate = Taxes Paid / Pre-Tax Income
2. If unavailable: Tax Expense / Pre-Tax Income
3. If unavailable: Statutory Rate (e.g., 35% US, 34% Brazil)
```

**Why Cash Tax Rate**: Accrual tax expense can differ significantly from cash taxes due to deferred tax assets/liabilities.

---

## Country Risk Adjustments

### Country Risk Premium

**Definition**: Additional return required to compensate for country-specific risks.

**Components**:
- Sovereign default risk
- Political instability
- Currency devaluation risk
- Regulatory uncertainty

**Typical Premiums** (as of 2025):
- **Argentina**: 8% (high inflation, FX controls)
- **Brazil**: 4% (moderate macro volatility)
- **Mexico**: 2% (relatively stable)
- **Chile**: 1% (OECD-like stability)

**Application**:
```
WACC_country = Base WACC + Country Risk Premium
```

---

## Thresholds & Benchmarks

### Inflation Regime Thresholds

- **High**: >40% YoY (hyperinflation territory)
- **Moderate**: 15-40% YoY (elevated inflation)
- **Normal**: 2-15% YoY (typical EM range)
- **Low**: <2% YoY (deflation risk)

### FX Volatility Thresholds

- **High**: >15% annualized std dev
- **Moderate**: 8-15% annualized std dev
- **Normal**: 3-8% annualized std dev
- **Low**: <3% annualized std dev

### Real Rate Thresholds

- **Positive**: Real rate > 0% (normal)
- **Mildly negative**: -5% to 0% (financial repression)
- **Severely negative**: <-5% (crisis/hyperinflation)

---

## Glossary of Abbreviations

| Term | Full Name | Description |
|------|-----------|-------------|
| ROIC | Return on Invested Capital | Core profitability metric |
| WACC | Weighted Average Cost of Capital | Hurdle rate for investments |
| NOPAT | Net Operating Profit After Tax | Operating earnings after tax |
| TTM | Trailing Twelve Months | Rolling 12-month sum |
| YoY | Year-over-Year | Annual growth rate |
| QoQ | Quarter-over-Quarter | Sequential quarterly growth |
| EP | Economic Profit | Profit above cost of capital |
| IC | Invested Capital | Capital employed in operations |
| NWC | Net Working Capital | Current assets - current liabilities |
| PPE | Property, Plant & Equipment | Fixed assets |
| CapEx | Capital Expenditure | Investment in fixed assets |
| OpEx | Operating Expenditure | Period expenses (R&D, S&M, etc.) |
| R&D | Research & Development | Innovation investment |
| S&M | Sales & Marketing | Customer acquisition investment |
| CPI | Consumer Price Index | Inflation measure |
| SELIC | Sistema Especial de Liquidação e Custódia | Brazil policy rate |
| BCB | Banco Central do Brasil | Brazil Central Bank |
| FX | Foreign Exchange | Currency markets |
| YTD | Year-to-Date | Cumulative from start of fiscal year |
| HP Filter | Hodrick-Prescott Filter | Trend/cycle decomposition |
| LATAM | Latin America | Geographic region |

---

## References

### Theoretical Frameworks

1. **VP/HOLT Methodology**: Credit Suisse HOLT
   - Capitalizes intangible investments
   - Economic (not accounting) profit focus
   - Long-term value creation orientation

2. **Capital Cycle Framework**: Marathon Asset Management
   - Supply-side investing framework
   - Focus on capital flows and competitive dynamics
   - Contrarian positioning based on capital scarcity

3. **Economic Profit**: Stern Stewart & Co.
   - EVA (Economic Value Added) framework
   - Focus on value creation vs destruction
   - Alignment of managerial incentives with shareholder value

### Data Sources

- **SEC EDGAR**: U.S. company financial filings (XBRL format)
- **Brazil Central Bank (BCB)**: Macro indicators via SGS API
- **Company Reports**: Quarterly 10-Q and annual 10-K filings

---

**Last Updated**: February 2026
**Version**: 1.0.0
