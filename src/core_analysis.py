#!/usr/bin/env python3
"""
Core analysis pipeline for AI-in-Education industry research.
Analyzes 教育产品统计_V5.csv and outputs JSON statistics for visualization.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = "/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv"
OUTPUT_DIR = "/Users/sakai/Desktop/产业调研/ai-edu-research/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load & Clean
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV with proper encoding and rename duplicate columns."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    # The CSV has 31 columns; column index 10 is unnamed/empty.
    # Duplicate names: 关键技术路径(8,23), 技术要素(9,24),
    #                  产业应用现状(6,25), 优势和创新点(7,26)
    # Columns 6-9 are case-level summaries; 23-26 are structured versions.
    # Rename to disambiguate.
    cols = list(df.columns)
    # pandas auto-renames duplicates with .1 suffix; verify and standardize
    rename_map = {}
    for c in cols:
        if c.endswith(".1"):
            base = c[:-2]
            rename_map[c] = f"{base}_结构化"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Drop the unnamed empty column if present
    unnamed_cols = [c for c in df.columns if c == "" or c.startswith("Unnamed")]
    df.drop(columns=unnamed_cols, inplace=True, errors="ignore")

    # Strip whitespace from all string cells
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Replace common missing-value markers with NaN
    df.replace(["", "nan", "NaN", "null", "NULL", "无", "未知", "N/A", "n/a"],
               np.nan, inplace=True)

    return df


def normalize_stage(s: str) -> str:
    """Normalize 学段 values to canonical forms."""
    if pd.isna(s):
        return "未知"
    s = s.strip()
    mapping = {
        "小学": "小学",
        "初中": "初中",
        "高中": "高中",
        "中学": "中学",
        "幼儿园": "幼儿园",
        "学前": "幼儿园",
        "职高": "职业高中",
        "中职": "职业高中",
        "职业高中": "职业高中",
    }
    for key, val in mapping.items():
        if key in s:
            return val
    return s


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning transformations."""
    df["学段_标准"] = df["学段"].apply(normalize_stage)
    # Ensure 案例编号 is string for grouping
    df["案例编号"] = df["案例编号"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 2. Analysis Functions
# ---------------------------------------------------------------------------
def save_json(data, name: str):
    """Save analysis result as JSON."""
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def value_counts_dict(series: pd.Series, top_n: int = None) -> dict:
    """Return {value: count} from a Series, optionally top N."""
    vc = series.dropna().value_counts()
    if top_n:
        vc = vc.head(top_n)
    return vc.to_dict()


def case_level(df: pd.DataFrame) -> pd.DataFrame:
    """Get one row per case (first row of each 案例编号)."""
    return df.drop_duplicates(subset="案例编号", keep="first")


# --- Province ---
def analyze_province(df: pd.DataFrame, cases: pd.DataFrame) -> dict:
    result = {
        "case_count_by_province": value_counts_dict(cases["省份"]),
        "tool_count_by_province": value_counts_dict(df["省份"]),
    }
    save_json(result, "province_distribution")
    return result


# --- School Stage ---
def analyze_stage(df: pd.DataFrame, cases: pd.DataFrame) -> dict:
    result = {
        "case_count_by_stage": value_counts_dict(cases["学段_标准"]),
        "tool_count_by_stage": value_counts_dict(df["学段_标准"]),
    }
    save_json(result, "stage_distribution")
    return result


# --- Subject ---
def analyze_subject(df: pd.DataFrame, cases: pd.DataFrame) -> dict:
    result = {
        "case_count_by_subject": value_counts_dict(cases["学科"]),
        "tool_count_by_subject": value_counts_dict(df["学科"]),
        "top20_subjects": value_counts_dict(cases["学科"], top_n=20),
    }
    save_json(result, "subject_distribution")
    return result


# --- Tool / Product ---
def analyze_tools(df: pd.DataFrame) -> dict:
    tool_col = "工具标准名" if "工具标准名" in df.columns else "工具名称"
    result = {
        "total_tool_mentions": int(df[tool_col].notna().sum()),
        "unique_tools": int(df[tool_col].nunique()),
        "top30_tools": value_counts_dict(df[tool_col], top_n=30),
        "product_form": value_counts_dict(df["产品形态"]),
        "product_category": value_counts_dict(df["产品分类"]),
    }
    save_json(result, "tool_product_distribution")
    return result

# --- Company Market Share ---
def analyze_companies(df: pd.DataFrame) -> dict:
    company_col = "公司" if "公司" in df.columns else "企业名称"
    vc = df[company_col].dropna().value_counts()
    total = int(vc.sum())
    top20 = vc.head(20)
    result = {
        "unique_companies": int(df[company_col].nunique()),
        "total_mentions": total,
        "top20_companies": top20.to_dict(),
        "top20_market_share_pct": (top20 / total * 100).round(2).to_dict(),
        "company_concentration_top5_pct": round(float(vc.head(5).sum() / total * 100), 2),
        "company_concentration_top10_pct": round(float(vc.head(10).sum() / total * 100), 2),
    }
    save_json(result, "company_market_share")
    return result


# --- Application Scenarios ---
def analyze_scenarios(df: pd.DataFrame) -> dict:
    result = {
        "primary_scenario": value_counts_dict(df["主要应用场景"], top_n=30),
        "secondary_scenario": value_counts_dict(df["次要应用场景"], top_n=30),
        "scenario_level1": value_counts_dict(df["应用场景（一级）"]),
        "scenario_level2": value_counts_dict(df["应用场景（二级）"], top_n=30),
        "cultivation_direction": value_counts_dict(df["培养方向"]),
    }
    save_json(result, "scenario_analysis")
    return result


# --- Self-developed vs Third-party ---
def analyze_self_developed(df: pd.DataFrame) -> dict:
    col = "是否自主研发"
    vc = df[col].dropna().value_counts()
    total = int(vc.sum())
    result = {
        "distribution": vc.to_dict(),
        "self_developed_ratio_pct": round(
            float(vc.get("TRUE", vc.get("是", 0))) / total * 100, 2
        ) if total > 0 else 0,
    }
    save_json(result, "self_developed_ratio")
    return result


# --- Technology Elements ---
def analyze_tech_elements(df: pd.DataFrame) -> dict:
    """Parse technology elements from the 技术要素 column (JSON arrays or text)."""
    tech_counter = Counter()
    col = "技术要素"
    for val in df[col].dropna():
        val = str(val).strip()
        # Try JSON array first
        if val.startswith("["):
            try:
                items = json.loads(val)
                for item in items:
                    tech_counter[item.strip()] += 1
                continue
            except json.JSONDecodeError:
                pass
        # Fallback: split by common delimiters
        for item in re.split(r"[,，、;\n]", val):
            item = item.strip().strip('"').strip("'")
            if len(item) > 1 and len(item) < 30:
                tech_counter[item] += 1

    top50 = dict(tech_counter.most_common(50))
    result = {
        "unique_tech_elements": len(tech_counter),
        "top50_tech_elements": top50,
    }
    save_json(result, "tech_elements")
    return result


# --- Industry Maturity ---
def analyze_industry_maturity(cases: pd.DataFrame) -> dict:
    """Extract maturity keywords from 产业应用现状."""
    col = "产业应用现状"
    maturity_keywords = {
        "初期": 0, "探索": 0, "起步": 0,
        "发展": 0, "成长": 0,
        "成熟": 0, "规模化": 0,
        "深度融合": 0, "常态化": 0,
    }
    for val in cases[col].dropna():
        val = str(val)
        for kw in maturity_keywords:
            if kw in val:
                maturity_keywords[kw] += 1
    result = {
        "maturity_keyword_freq": maturity_keywords,
        "sample_descriptions": cases[col].dropna().head(10).tolist(),
    }
    save_json(result, "industry_maturity")
    return result

# --- Cross-tabulations ---
def analyze_cross_tabs(df: pd.DataFrame, cases: pd.DataFrame) -> dict:
    """Stage x Subject, Stage x Scenario, Province x Stage cross-tabs."""

    def crosstab_to_dict(index_col, columns_col, data, top_n_cols=15):
        ct = pd.crosstab(data[index_col], data[columns_col])
        # Keep only top N columns by total
        top_cols = ct.sum().nlargest(top_n_cols).index.tolist()
        ct = ct[top_cols]
        return {
            "index": ct.index.tolist(),
            "columns": ct.columns.tolist(),
            "data": ct.values.tolist(),
        }

    result = {}
    # Stage x Subject (case level)
    result["stage_subject"] = crosstab_to_dict("学段_标准", "学科", cases, 15)
    # Stage x Scenario L1 (row level)
    result["stage_scenario_l1"] = crosstab_to_dict(
        "学段_标准", "应用场景（一级）", df, 10
    )
    # Province x Stage (case level)
    result["province_stage"] = crosstab_to_dict("省份", "学段_标准", cases, 10)

    save_json(result, "cross_tabulations")
    return result


# ---------------------------------------------------------------------------
# 3. Summary Report
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, cases: pd.DataFrame, results: dict):
    """Print a concise summary report to stdout."""
    print("=" * 70)
    print("  AI-in-Education Industry Research -- Data Analysis Summary")
    print("=" * 70)
    print(f"\nTotal rows:           {len(df):,}")
    print(f"Unique cases:         {cases.shape[0]:,}")
    print(f"Columns:              {len(df.columns)}")
    print(f"Unique tools:         {results['tools']['unique_tools']:,}")
    print(f"Unique companies:     {results['companies']['unique_companies']:,}")

    print("\n--- Cases by School Stage ---")
    for k, v in sorted(
        results["stage"]["case_count_by_stage"].items(), key=lambda x: -x[1]
    ):
        print(f"  {k:12s}  {v:5d}")

    print("\n--- Top 10 Provinces (by case count) ---")
    for i, (k, v) in enumerate(
        sorted(
            results["province"]["case_count_by_province"].items(),
            key=lambda x: -x[1],
        )
    ):
        if i >= 10:
            break
        print(f"  {k:12s}  {v:5d}")

    print("\n--- Top 10 Tools ---")
    for i, (k, v) in enumerate(results["tools"]["top30_tools"].items()):
        if i >= 10:
            break
        print(f"  {k:20s}  {v:5d}")

    print("\n--- Top 10 Companies ---")
    for i, (k, v) in enumerate(results["companies"]["top20_companies"].items()):
        if i >= 10:
            break
        pct = results["companies"]["top20_market_share_pct"].get(k, 0)
        print(f"  {k:20s}  {v:5d}  ({pct:.1f}%)")

    print(f"\n  Top-5 concentration:  "
          f"{results['companies']['company_concentration_top5_pct']:.1f}%")
    print(f"  Top-10 concentration: "
          f"{results['companies']['company_concentration_top10_pct']:.1f}%")

    print("\n--- Product Form ---")
    for k, v in sorted(
        results["tools"]["product_form"].items(), key=lambda x: -x[1]
    ):
        print(f"  {k:24s}  {v:5d}")

    print("\n--- Application Scenario (Level 1) ---")
    for k, v in sorted(
        results["scenarios"]["scenario_level1"].items(), key=lambda x: -x[1]
    ):
        print(f"  {k:16s}  {v:5d}")

    print("\n--- Self-developed Ratio ---")
    for k, v in results["self_dev"]["distribution"].items():
        print(f"  {k:10s}  {v:5d}")
    print(f"  Self-developed: {results['self_dev']['self_developed_ratio_pct']:.1f}%")

    print("\n--- Top 15 Technology Elements ---")
    for i, (k, v) in enumerate(results["tech"]["top50_tech_elements"].items()):
        if i >= 15:
            break
        print(f"  {k:30s}  {v:5d}")

    print("\n--- Industry Maturity Keywords ---")
    for k, v in sorted(
        results["maturity"]["maturity_keyword_freq"].items(), key=lambda x: -x[1]
    ):
        if v > 0:
            print(f"  {k:12s}  {v:5d}")

    print("\n--- Top 10 Subjects ---")
    for i, (k, v) in enumerate(
        sorted(results["subject"]["case_count_by_subject"].items(), key=lambda x: -x[1])
    ):
        if i >= 10:
            break
        print(f"  {k:16s}  {v:5d}")

    print(f"\nAll JSON outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data(CSV_PATH)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("Cleaning data...")
    df = clean_data(df)
    cases = case_level(df)
    print(f"  {len(cases):,} unique cases")

    print("Running analyses...\n")
    results = {}
    results["province"] = analyze_province(df, cases)
    results["stage"] = analyze_stage(df, cases)
    results["subject"] = analyze_subject(df, cases)
    results["tools"] = analyze_tools(df)
    results["companies"] = analyze_companies(df)
    results["scenarios"] = analyze_scenarios(df)
    results["self_dev"] = analyze_self_developed(df)
    results["tech"] = analyze_tech_elements(df)
    results["maturity"] = analyze_industry_maturity(cases)
    results["cross_tabs"] = analyze_cross_tabs(df, cases)

    # Save a combined metadata file
    meta = {
        "total_rows": len(df),
        "unique_cases": len(cases),
        "columns": list(df.columns),
        "output_files": [
            "province_distribution.json",
            "stage_distribution.json",
            "subject_distribution.json",
            "tool_product_distribution.json",
            "company_market_share.json",
            "scenario_analysis.json",
            "self_developed_ratio.json",
            "tech_elements.json",
            "industry_maturity.json",
            "cross_tabulations.json",
        ],
    }
    save_json(meta, "metadata")

    print_summary(df, cases, results)


if __name__ == "__main__":
    main()
