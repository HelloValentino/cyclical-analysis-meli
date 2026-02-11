#!/usr/bin/env python3
"""Check FP distribution for current and alternative XBRL tags."""
import json, sys
sys.path.insert(0, "src")
from sec_data_extractor import extract_metric_facts

with open("data/raw/sec_cache/meli_companyfacts.json") as f:
    cf = json.load(f)

check = {
    "accum_dep CURRENT": "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
    "accum_dep LEASE-ERA": "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAccumulatedDepreciationAndAmortization",
    "accum_dep BEFORE": "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization",
    "accum_dep AFTER": "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    "accum_dep OTHER": "PropertyPlantAndEquipmentOtherAccumulatedDepreciation",
    "lt_debt CURRENT": "LongTermDebt",
    "lt_debt W/LEASES": "LongTermDebtAndCapitalLeaseObligations",
    "lt_debt LOANS": "LongTermLoansPayable",
    "lt_debt NOTES_BANK_NC": "NotesPayableToBankNoncurrent",
    "credit NotesRecNet": "NotesReceivableNet",
    "credit LoansLeasesNet": "LoansAndLeasesReceivableNetReportedAmount",
    "credit NotesLoansNetCurr": "NotesAndLoansReceivableNetCurrent",
    "credit NotesLoansNetNonCurr": "NotesAndLoansReceivableNetNoncurrent",
    "credit FinRecvAfterLoss": "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
    "credit LoansLeasesGross": "LoansAndLeasesReceivableGrossCarryingAmount",
    "taxes_paid CURRENT": "IncomeTaxesPaid",
    "taxes_paid NET": "IncomeTaxesPaidNet",
}

for label, tag in check.items():
    df = extract_metric_facts(cf, tag)
    if df.empty:
        print(f"{label:40s} -> NO DATA")
        continue
    fp_counts = dict(df["fp"].value_counts())
    yr_range = f"{str(df['end'].min())[:10]}..{str(df['end'].max())[:10]}"
    print(f"{label:40s} {len(df):3d} rows  FP={fp_counts}  range={yr_range}")

