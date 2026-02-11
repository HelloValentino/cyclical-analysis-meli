"""Deep dive part 2: verify standalone vs YTD and Q4 inference potential."""
import json, sys
sys.path.insert(0, "src")
import pandas as pd, numpy as np
from sec_data_extractor import extract_metric_facts, METRICS, _looks_ytd

with open("data/raw/sec_cache/meli_companyfacts.json") as f:
    facts = json.load(f)

concepts = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueServicesNet",
]

# 1) For RevenueFromContract..., check if values are standalone or YTD
# by looking at start dates and comparing Q1+Q2+Q3 vs FY
print("=" * 80)
print("1) STANDALONE vs YTD ANALYSIS (RevenueFromContract...)")
print("=" * 80)
c = "RevenueFromContractWithCustomerExcludingAssessedTax"
entries = facts["facts"]["us-gaap"][c]["units"]["USD"]
for fy in range(2019, 2026):
    fy_e = [e for e in entries if e.get("fy") == fy]
    by_fp = {}
    for e in sorted(fy_e, key=lambda x: x.get("filed", "")):
        by_fp[e["fp"]] = e
    vals = {fp: e["val"] for fp, e in by_fp.items()}
    starts = {fp: e.get("start") for fp, e in by_fp.items()}

    q1, q2, q3 = vals.get("Q1"), vals.get("Q2"), vals.get("Q3")
    fy_val = vals.get("FY")

    print(f"\n  FY{fy}:")
    for fp in ["Q1", "Q2", "Q3", "FY"]:
        if fp in vals:
            s = starts.get(fp, "?")
            e = by_fp[fp].get("end", "?")
            print(f"    {fp}: start={s} end={e} val={vals[fp]:>15,.0f}")

    if q1 and q2 and q3:
        raw_sum = q1 + q2 + q3
        ytd_check = _looks_ytd(q1, q2, q3)
        print(f"    _looks_ytd({q1:,.0f}, {q2:,.0f}, {q3:,.0f}) = {ytd_check}")
        print(f"    Raw Q1+Q2+Q3 = {raw_sum:,.0f}")
        if fy_val:
            print(f"    FY = {fy_val:,.0f}")
            implied_q4_standalone = fy_val - raw_sum
            print(f"    Implied Q4 (if standalone) = FY - sum = {implied_q4_standalone:,.0f}")
            if ytd_check:
                # If YTD: standalone = Q1, Q2-Q1, Q3-Q2
                sq1, sq2, sq3 = q1, q2 - q1, q3 - q2
                implied_q4_ytd = fy_val - q3  # FY - Q3_ytd
                print(f"    Standalone from YTD: Q1={sq1:,.0f} Q2={sq2:,.0f} Q3={sq3:,.0f}")
                print(f"    Implied Q4 (if YTD) = FY - Q3_ytd = {implied_q4_ytd:,.0f}")

# 2) Check Revenues concept for the same analysis
print("\n" + "=" * 80)
print("2) STANDALONE vs YTD ANALYSIS (Revenues)")
print("=" * 80)
c = "Revenues"
entries = facts["facts"]["us-gaap"][c]["units"]["USD"]
for fy in range(2017, 2026):
    fy_e = [e for e in entries if e.get("fy") == fy]
    by_fp = {}
    for e in sorted(fy_e, key=lambda x: x.get("filed", "")):
        by_fp[e["fp"]] = e
    vals = {fp: e["val"] for fp, e in by_fp.items()}
    starts = {fp: e.get("start") for fp, e in by_fp.items()}

    q1, q2, q3 = vals.get("Q1"), vals.get("Q2"), vals.get("Q3")
    fy_val = vals.get("FY")

    if not any([q1, q2, q3, fy_val]):
        continue

    print(f"\n  FY{fy}:")
    for fp in ["Q1", "Q2", "Q3", "FY"]:
        if fp in vals:
            s = starts.get(fp, "?")
            e_date = by_fp[fp].get("end", "?")
            print(f"    {fp}: start={s} end={e_date} val={vals[fp]:>15,.0f}")

    if q1 and q2 and q3:
        ytd_check = _looks_ytd(q1, q2, q3)
        print(f"    _looks_ytd = {ytd_check}")
        if fy_val:
            raw_sum = q1 + q2 + q3
            print(f"    Raw Q1+Q2+Q3 = {raw_sum:,.0f}, FY = {fy_val:,.0f}")

# 3) SalesRevenueServicesNet for 2015-2017
print("\n" + "=" * 80)
print("3) SalesRevenueServicesNet (2015-2017)")
print("=" * 80)
c = "SalesRevenueServicesNet"
entries = facts["facts"]["us-gaap"][c]["units"]["USD"]
for fy in range(2015, 2018):
    fy_e = [e for e in entries if e.get("fy") == fy]
    by_fp = {}
    for e in sorted(fy_e, key=lambda x: x.get("filed", "")):
        by_fp[e["fp"]] = e
    vals = {fp: e["val"] for fp, e in by_fp.items()}
    starts = {fp: e.get("start") for fp, e in by_fp.items()}

    print(f"\n  FY{fy}:")
    for fp in ["Q1", "Q2", "Q3", "FY"]:
        if fp in vals:
            s = starts.get(fp, "?")
            e_date = by_fp[fp].get("end", "?")
            print(f"    {fp}: start={s} end={e_date} val={vals[fp]:>15,.0f}")

