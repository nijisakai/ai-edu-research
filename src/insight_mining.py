#!/usr/bin/env python3
"""
Deep Insight Mining: 从数据中提取真金
Comprehensive statistical analysis with counter-intuitive findings,
digital divide indices, tool ecosystem networks, and policy implications.
"""

import json
import math
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "output"
CSV_PATH = OUTPUT / "教育产品统计_V6_框架标注.csv"

# Province economic tiers (GDP per capita ranking 2024, simplified)
PROVINCE_ECON_TIER = {
    "北京市": 1, "上海市": 1, "江苏省": 1, "浙江省": 1, "广东省": 1,
    "天津市": 2, "福建省": 2, "山东省": 2, "湖北省": 2, "重庆市": 2,
    "安徽省": 3, "湖南省": 3, "四川省": 3, "辽宁省": 3, "河北省": 3,
    "陕西省": 3, "江西省": 3, "河南省": 3, "山西省": 3,
    "内蒙古自治区": 3, "吉林省": 3, "黑龙江省": 3, "海南省": 3,
    "宁夏回族自治区": 4, "新疆维吾尔自治区": 4, "甘肃省": 4,
    "贵州省": 4, "青海省": 4,
}

# 五育 mapping
WUYU_MAP = {
    "智育": "智育", "德育": "德育", "美育": "美育", "体育": "体育", "劳育": "劳育",
    "德智体美劳": "综合",
}

def normalize_wuyu(val):
    """Map 培养方向 to standard 五育 category."""
    if pd.isna(val):
        return "未提及"
    val = str(val).strip()
    for key in WUYU_MAP:
        if key in val:
            return WUYU_MAP[key]
    if "劳" in val:
        return "劳育"
    if "体" in val:
        return "体育"
    if "美" in val:
        return "美育"
    if "德" in val:
        return "德育"
    if "智" in val:
        return "智育"
    return val


def gini_coefficient(values):
    """Calculate Gini coefficient for a list of values."""
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def hhi_index(shares):
    """Calculate Herfindahl-Hirschman Index from counts."""
    total = sum(shares)
    if total == 0:
        return 0.0
    return sum((s / total * 100) ** 2 for s in shares)


def effect_size_cramers_v(contingency_table):
    """Calculate Cramer's V effect size from a contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return math.sqrt(chi2 / (n * min_dim))


# ============================================================
# 1. Counter-intuitive Findings
# ============================================================

def analyze_counter_intuitive(df):
    """Find patterns that defy expectations."""
    findings = []

    # 1a. Provinces with higher innovation than expected by economic tier
    prov_stats = df.groupby("省份").agg(
        mean_innovation=("创新深度评分", "mean"),
        mean_gen=("gen_num", "mean"),
        mean_istar=("istar_num", "mean"),
        count=("案例编号", "count"),
    ).reset_index()
    prov_stats["econ_tier"] = prov_stats["省份"].map(PROVINCE_ECON_TIER)
    prov_stats = prov_stats.dropna(subset=["econ_tier"])
    prov_stats = prov_stats[prov_stats["count"] >= 10]

    # Correlation: economic tier vs innovation
    if len(prov_stats) >= 5:
        r, p = stats.spearmanr(prov_stats["econ_tier"], prov_stats["mean_innovation"])
        findings.append({
            "type": "province_econ_vs_innovation",
            "finding": f"经济水平与创新深度的Spearman相关系数 r={r:.3f}, p={p:.4f}",
            "interpretation": "正值表示经济越好创新越高，负值表示反直觉" if r > 0
                else "经济水平与创新深度呈负相关——欠发达地区反而创新更深",
            "r": round(r, 4), "p": round(p, 4),
        })

    # Identify overperformers: tier 3-4 but top innovation
    tier34 = prov_stats[prov_stats["econ_tier"] >= 3].sort_values("mean_innovation", ascending=False)
    if len(tier34) > 0:
        top_overperformer = tier34.iloc[0]
        findings.append({
            "type": "overperformer_province",
            "finding": f"经济欠发达但创新领先: {top_overperformer['省份']} "
                       f"(经济tier={int(top_overperformer['econ_tier'])}, "
                       f"创新均分={top_overperformer['mean_innovation']:.2f})",
            "all_overperformers": tier34[["省份", "econ_tier", "mean_innovation", "mean_gen"]].to_dict("records"),
        })

    # 1b. Stage surprises: 幼儿园 vs 高中 tech generation
    stage_gen = df[df["学段"].isin(["幼儿园", "小学", "初中", "高中"])].groupby("学段").agg(
        mean_gen=("gen_num", "mean"),
        pct_gen4plus=("gen_num", lambda x: (x >= 4).mean() * 100),
        mean_innovation=("创新深度评分", "mean"),
        mean_istar=("istar_num", "mean"),
    ).reset_index()
    stage_gen = stage_gen.sort_values("mean_gen", ascending=False)
    findings.append({
        "type": "stage_tech_generation",
        "finding": f"技术代际最高学段: {stage_gen.iloc[0]['学段']} "
                   f"(均值={stage_gen.iloc[0]['mean_gen']:.2f}), "
                   f"最低: {stage_gen.iloc[-1]['学段']} "
                   f"(均值={stage_gen.iloc[-1]['mean_gen']:.2f})",
        "detail": stage_gen.to_dict("records"),
    })

    # 1c. Low-tech beating high-tech
    gen12 = df[df["gen_num"] <= 2]["创新深度评分"]
    gen45 = df[df["gen_num"] >= 4]["创新深度评分"]
    if len(gen12) > 0 and len(gen45) > 0:
        u_stat, u_p = stats.mannwhitneyu(gen12, gen45, alternative="two-sided")
        high_low_tech = df[df["gen_num"] <= 2].groupby("创新深度评分").size()
        pct_high_innov_lowtech = (df[(df["gen_num"] <= 2) & (df["创新深度评分"] >= 4)].shape[0]
                                   / max(df[df["gen_num"] <= 2].shape[0], 1) * 100)
        findings.append({
            "type": "low_tech_high_innovation",
            "finding": f"Gen1-2(低技术)中有{pct_high_innov_lowtech:.1f}%达到创新深度4-5分",
            "gen12_mean": round(gen12.mean(), 3),
            "gen45_mean": round(gen45.mean(), 3),
            "mann_whitney_U": round(u_stat, 1),
            "p_value": round(u_p, 6),
        })

    # 1d. Subject surprises
    subj_stats = df.groupby("学科").agg(
        mean_innovation=("创新深度评分", "mean"),
        mean_gen=("gen_num", "mean"),
        count=("案例编号", "count"),
    ).reset_index()
    subj_stats = subj_stats[subj_stats["count"] >= 15].sort_values("mean_innovation", ascending=False)
    findings.append({
        "type": "subject_innovation_ranking",
        "finding": f"创新深度最高学科: {subj_stats.iloc[0]['学科']} "
                   f"({subj_stats.iloc[0]['mean_innovation']:.2f}), "
                   f"最低: {subj_stats.iloc[-1]['学科']} "
                   f"({subj_stats.iloc[-1]['mean_innovation']:.2f})",
        "ranking": subj_stats[["学科", "mean_innovation", "mean_gen", "count"]].to_dict("records"),
    })

    return findings


# ============================================================
# 2. Digital Divide Deep Analysis
# ============================================================

def analyze_digital_divide(df):
    """Calculate digital divide indices across provinces."""
    result = {}

    # 2a. 数字鸿沟指数 per province
    prov_groups = df.groupby("省份")
    divide_indices = {}
    for prov, grp in prov_groups:
        if len(grp) < 5:
            continue
        gen45 = grp[grp["gen_num"] >= 4].shape[0]
        gen12 = grp[grp["gen_num"] <= 2].shape[0]
        divide_indices[prov] = {
            "digital_divide_index": round(gen45 / max(gen12, 1), 3),
            "gen4plus_pct": round(gen45 / len(grp) * 100, 1),
            "gen12_pct": round(gen12 / len(grp) * 100, 1),
            "mean_innovation": round(grp["创新深度评分"].mean(), 2),
            "mean_istar": round(grp["istar_num"].mean(), 2),
            "econ_tier": PROVINCE_ECON_TIER.get(prov, None),
            "n": len(grp),
        }
    result["province_divide_index"] = divide_indices

    # 2b. Correlation: econ tier vs HM2C ratio
    prov_df = pd.DataFrame(divide_indices).T
    prov_df = prov_df.dropna(subset=["econ_tier"])
    if len(prov_df) >= 5:
        r, p = stats.spearmanr(prov_df["econ_tier"], prov_df["mean_istar"])
        result["econ_vs_istar_correlation"] = {
            "spearman_r": round(r, 4), "p_value": round(p, 4),
            "interpretation": "经济水平与人机协同层级的关系",
        }

    # 2c. 跨越式发展 provinces
    leapfrog = []
    for prov, info in divide_indices.items():
        tier = info.get("econ_tier")
        if tier and tier >= 3 and info["digital_divide_index"] > 1.0:
            leapfrog.append({"province": prov, **info})
    leapfrog.sort(key=lambda x: x["digital_divide_index"], reverse=True)
    result["leapfrog_provinces"] = leapfrog

    # 2d. Gini coefficient of tool diversity
    prov_tool_counts = df.groupby("省份")["工具标准名"].nunique()
    gini = gini_coefficient(prov_tool_counts.values)
    result["tool_diversity_gini"] = round(gini, 4)
    result["tool_diversity_gini_interpretation"] = (
        "0=完全均等, 1=完全集中. "
        f"当前Gini={gini:.4f}, {'高度不均' if gini > 0.5 else '中度不均' if gini > 0.3 else '相对均衡'}"
    )

    # Gini of innovation depth
    prov_innov = df.groupby("省份")["创新深度评分"].mean().dropna()
    gini_innov = gini_coefficient(prov_innov.values)
    result["innovation_gini"] = round(gini_innov, 4)

    # Stage-level divide
    stage_divide = {}
    for stage in ["幼儿园", "小学", "初中", "高中"]:
        sg = df[df["学段"] == stage]
        if len(sg) < 5:
            continue
        gen45 = sg[sg["gen_num"] >= 4].shape[0]
        gen12 = sg[sg["gen_num"] <= 2].shape[0]
        stage_divide[stage] = {
            "digital_divide_index": round(gen45 / max(gen12, 1), 3),
            "gen4plus_pct": round(gen45 / len(sg) * 100, 1),
            "hm2c_pct": round((sg["istar_num"] == 2).mean() * 100, 1),
        }
    result["stage_divide"] = stage_divide

    return result


# ============================================================
# 3. Tool Ecosystem Network Analysis
# ============================================================

def analyze_tool_ecosystem(df):
    """Analyze tool co-occurrence and combination patterns."""
    result = {}

    # Group tools by case
    case_tools = df.groupby("案例编号")["工具标准名"].apply(list).to_dict()
    case_gen = df.groupby("案例编号")["产品技术代际"].apply(list).to_dict()
    case_innovation = df.groupby("案例编号")["创新深度评分"].mean().to_dict()

    # 3a. Co-occurrence matrix
    cooccurrence = Counter()
    for case_id, tools in case_tools.items():
        unique_tools = list(set(t for t in tools if pd.notna(t)))
        for t1, t2 in combinations(sorted(unique_tools), 2):
            cooccurrence[(t1, t2)] += 1

    top_pairs = cooccurrence.most_common(30)
    result["top_tool_pairs"] = [
        {"tool1": p[0][0], "tool2": p[0][1], "co_occurrences": p[1]}
        for p in top_pairs
    ]

    # 3b. Tool combination patterns (bundles)
    bundle_counter = Counter()
    bundle_innovation = defaultdict(list)
    for case_id, tools in case_tools.items():
        unique_tools = tuple(sorted(set(t for t in tools if pd.notna(t))))
        if len(unique_tools) >= 2:
            bundle_counter[unique_tools] += 1
            if case_id in case_innovation:
                bundle_innovation[unique_tools].append(case_innovation[case_id])

    top_bundles = bundle_counter.most_common(20)
    result["top_tool_bundles"] = [
        {
            "tools": list(b[0]),
            "frequency": b[1],
            "mean_innovation": round(np.mean(bundle_innovation.get(b[0], [0])), 2),
        }
        for b in top_bundles
    ]

    # 3c. Tool combinations vs innovation depth
    multi_tool_cases = {cid: tools for cid, tools in case_tools.items()
                        if len(set(t for t in tools if pd.notna(t))) >= 2}
    single_tool_cases = {cid: tools for cid, tools in case_tools.items()
                         if len(set(t for t in tools if pd.notna(t))) == 1}

    multi_innov = [case_innovation[c] for c in multi_tool_cases if c in case_innovation]
    single_innov = [case_innovation[c] for c in single_tool_cases if c in case_innovation]

    if multi_innov and single_innov:
        u_stat, u_p = stats.mannwhitneyu(multi_innov, single_innov, alternative="two-sided")
        result["multi_vs_single_tool"] = {
            "multi_tool_mean_innovation": round(np.mean(multi_innov), 3),
            "single_tool_mean_innovation": round(np.mean(single_innov), 3),
            "multi_tool_n": len(multi_innov),
            "single_tool_n": len(single_innov),
            "mann_whitney_U": round(u_stat, 1),
            "p_value": round(u_p, 6),
        }

    # 3d. Tool → scenario mapping
    tool_scenario = df.groupby("工具标准名")["应用场景（一级）"].apply(
        lambda x: x.value_counts().head(3).to_dict()
    ).to_dict()
    # Top 15 tools
    top_tools = df["工具标准名"].value_counts().head(15).index.tolist()
    result["tool_scenario_map"] = {t: tool_scenario.get(t, {}) for t in top_tools}

    # 3e. Product category co-occurrence (more meaningful than individual tools)
    case_cats = df.groupby("案例编号")["产品分类"].apply(lambda x: list(set(x.dropna())))
    cat_cooccurrence = Counter()
    for case_id, cats in case_cats.items():
        for c1, c2 in combinations(sorted(cats), 2):
            cat_cooccurrence[(c1, c2)] += 1
    result["product_category_cooccurrence"] = [
        {"cat1": p[0][0], "cat2": p[0][1], "co_occurrences": p[1]}
        for p in cat_cooccurrence.most_common(15)
    ]

    # 3f. Tech path co-occurrence
    case_tech = df.groupby("案例编号")["技术路径类型"].apply(lambda x: list(set(x.dropna())))
    tech_cooccurrence = Counter()
    for case_id, techs in case_tech.items():
        for t1, t2 in combinations(sorted(techs), 2):
            tech_cooccurrence[(t1, t2)] += 1
    result["tech_path_cooccurrence"] = [
        {"path1": p[0][0], "path2": p[0][1], "co_occurrences": p[1]}
        for p in tech_cooccurrence.most_common(15)
    ]

    # 3g. Tool diversity per case vs innovation (correlation)
    case_tool_count = df.groupby("案例编号")["工具标准名"].nunique()
    case_innov_mean = df.groupby("案例编号")["创新深度评分"].mean()
    merged = pd.DataFrame({"n_tools": case_tool_count, "innovation": case_innov_mean}).dropna()
    if len(merged) > 10:
        r, p = stats.spearmanr(merged["n_tools"], merged["innovation"])
        result["tool_diversity_innovation_corr"] = {
            "spearman_r": round(r, 4), "p_value": round(p, 6),
            "interpretation": f"工具多样性与创新深度{'正相关' if r > 0 else '负相关'} (r={r:.3f})",
        }

    return result


# ============================================================
# 4. Teacher Innovation Behavior Patterns
# ============================================================

def analyze_teacher_innovation(df):
    """Profile teacher innovation behaviors."""
    result = {}

    # 4a. Self-developed tools profile
    self_dev = df[df["是否自主研发"].astype(str).str.upper().isin(["TRUE", "True", "true", "1"])]
    non_self = df[~df["是否自主研发"].astype(str).str.upper().isin(["TRUE", "True", "true", "1"])]

    result["self_developed"] = {
        "count": len(self_dev),
        "pct_of_total": round(len(self_dev) / len(df) * 100, 1),
        "mean_innovation": round(self_dev["创新深度评分"].mean(), 2) if len(self_dev) > 0 else None,
        "non_self_mean_innovation": round(non_self["创新深度评分"].mean(), 2),
        "top_provinces": self_dev["省份"].value_counts().head(5).to_dict(),
        "top_stages": self_dev["学段"].value_counts().head(5).to_dict(),
        "top_subjects": self_dev["学科"].value_counts().head(5).to_dict(),
    }

    # Statistical test: self-dev vs non-self innovation
    if len(self_dev) > 5 and len(non_self) > 5:
        u, p = stats.mannwhitneyu(
            self_dev["创新深度评分"].dropna(),
            non_self["创新深度评分"].dropna(),
            alternative="two-sided"
        )
        result["self_developed"]["mann_whitney_p"] = round(p, 6)

    # 4b. Multi-tool users
    case_tool_count = df.groupby("案例编号")["工具标准名"].nunique()
    case_innovation = df.groupby("案例编号")["创新深度评分"].mean()
    merged = pd.DataFrame({"n_tools": case_tool_count, "innovation": case_innovation}).dropna()

    if len(merged) > 10:
        r, p = stats.spearmanr(merged["n_tools"], merged["innovation"])
        result["tool_count_vs_innovation"] = {
            "spearman_r": round(r, 4),
            "p_value": round(p, 6),
            "mean_tools_per_case": round(merged["n_tools"].mean(), 2),
            "interpretation": "工具数量与创新深度的关系",
        }

    # 4c. Top 5% innovators profile
    threshold_95 = df["创新深度评分"].quantile(0.95)
    top5 = df[df["创新深度评分"] >= threshold_95]
    bottom25 = df[df["创新深度评分"] <= df["创新深度评分"].quantile(0.25)]

    result["pioneers_top5pct"] = {
        "threshold": round(threshold_95, 1),
        "count": len(top5),
        "top_provinces": top5["省份"].value_counts().head(5).to_dict(),
        "top_stages": top5["学段"].value_counts().head(3).to_dict(),
        "top_subjects": top5["学科"].value_counts().head(5).to_dict(),
        "mean_gen": round(top5["gen_num"].mean(), 2),
        "hm2c_pct": round((top5["istar_num"] == 2).mean() * 100, 1),
        "top_tech_paths": top5["技术路径类型"].value_counts().head(3).to_dict(),
    }

    result["followers_bottom25pct"] = {
        "threshold": round(df["创新深度评分"].quantile(0.25), 1),
        "count": len(bottom25),
        "top_provinces": bottom25["省份"].value_counts().head(5).to_dict(),
        "top_stages": bottom25["学段"].value_counts().head(3).to_dict(),
        "mean_gen": round(bottom25["gen_num"].mean(), 2),
        "hm2c_pct": round((bottom25["istar_num"] == 2).mean() * 100, 1),
    }

    return result


# ============================================================
# 5. Subject-AI Fit Matrix
# ============================================================

def analyze_subject_ai_fit(df):
    """Build subject x AI compatibility matrix."""
    result = {}

    # Filter to subjects with enough data
    subj_counts = df["学科"].value_counts()
    valid_subjects = subj_counts[subj_counts >= 15].index.tolist()
    sdf = df[df["学科"].isin(valid_subjects)]

    matrix = []
    for subj in valid_subjects:
        sg = sdf[sdf["学科"] == subj]
        matrix.append({
            "subject": subj,
            "n": len(sg),
            "mean_istar": round(sg["istar_num"].mean(), 3),
            "mean_innovation": round(sg["创新深度评分"].mean(), 3),
            "tool_diversity": sg["工具标准名"].nunique(),
            "gen4plus_pct": round((sg["gen_num"] >= 4).mean() * 100, 1),
            "hm2c_pct": round((sg["istar_num"] == 2).mean() * 100, 1),
            "mean_gen": round(sg["gen_num"].mean(), 3),
            "d1_deep_learning_pct": round(sg["D1_深度学习"].mean() * 100, 1),
        })

    matrix.sort(key=lambda x: x["mean_innovation"], reverse=True)
    result["subject_matrix"] = matrix

    # AI-ready vs AI-resistant
    mat_df = pd.DataFrame(matrix)
    if len(mat_df) > 0:
        median_innov = mat_df["mean_innovation"].median()
        median_gen = mat_df["gen4plus_pct"].median()
        ai_ready = mat_df[(mat_df["mean_innovation"] >= median_innov) &
                          (mat_df["gen4plus_pct"] >= median_gen)]["subject"].tolist()
        ai_resistant = mat_df[(mat_df["mean_innovation"] < median_innov) &
                              (mat_df["gen4plus_pct"] < median_gen)]["subject"].tolist()
        result["ai_ready_subjects"] = ai_ready
        result["ai_resistant_subjects"] = ai_resistant

    # 五育 coverage analysis
    wuyu_stats = df.groupby("五育").agg(
        count=("案例编号", "count"),
        mean_innovation=("创新深度评分", "mean"),
        mean_gen=("gen_num", "mean"),
        hm2c_pct=("istar_num", lambda x: (x == 2).mean() * 100),
    ).reset_index()
    wuyu_stats = wuyu_stats.sort_values("count", ascending=False)
    result["wuyu_coverage"] = wuyu_stats.round(2).to_dict("records")

    # Chi-square: subject vs iSTAR level
    ct = pd.crosstab(sdf["学科"], sdf["iSTAR人机协同层级"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = effect_size_cramers_v(ct)
        result["subject_istar_chi2"] = {
            "chi2": round(chi2, 2), "p_value": round(p, 6),
            "dof": dof, "cramers_v": round(v, 4),
            "interpretation": f"学科与iSTAR层级{'显著相关' if p < 0.05 else '无显著关联'} "
                              f"(V={v:.3f}, {'大' if v > 0.3 else '中' if v > 0.1 else '小'}效应量)",
        }

    return result


# ============================================================
# 6. Temporal Inference (using tech generation as proxy)
# ============================================================

def analyze_temporal_inference(df):
    """Infer adoption timeline patterns from tech generation distribution."""
    result = {}

    # Technology adoption curve
    gen_dist = df["产品技术代际"].value_counts().sort_index()
    total = gen_dist.sum()
    gen_pcts = (gen_dist / total * 100).round(1).to_dict()
    result["tech_generation_distribution"] = gen_pcts

    # Cumulative adoption curve
    gen_order = ["Gen1_传统信息化", "Gen2_互联网+", "Gen3_AI辅助", "Gen4_大模型", "Gen5_多模态AI"]
    cumulative = {}
    running = 0
    for g in gen_order:
        running += gen_dist.get(g, 0)
        cumulative[g] = round(running / total * 100, 1)
    result["cumulative_adoption"] = cumulative

    # LLM tipping point: Gen4+Gen5 vs Gen1+Gen2+Gen3
    llm_count = df[df["gen_num"] >= 4].shape[0]
    pre_llm = df[df["gen_num"] <= 3].shape[0]
    result["llm_tipping_point"] = {
        "llm_era_pct": round(llm_count / total * 100, 1),
        "pre_llm_pct": round(pre_llm / total * 100, 1),
        "ratio": round(llm_count / max(pre_llm, 1), 3),
        "interpretation": "LLM时代工具已超过传统工具" if llm_count > pre_llm
            else "传统工具仍占多数，LLM尚未达到拐点",
    }

    # Scenario growth prediction: which scenarios have highest Gen4+ ratio
    scenario_gen = df.groupby("应用场景（一级）").agg(
        gen4plus_pct=("gen_num", lambda x: (x >= 4).mean() * 100),
        mean_gen=("gen_num", "mean"),
        count=("案例编号", "count"),
    ).reset_index()
    scenario_gen = scenario_gen[scenario_gen["count"] >= 10]
    scenario_gen = scenario_gen.sort_values("gen4plus_pct", ascending=False)
    result["scenario_growth_potential"] = scenario_gen.round(2).to_dict("records")

    # Stage adoption curve
    stage_gen = df[df["学段"].isin(["幼儿园", "小学", "初中", "高中"])].groupby("学段").agg(
        gen4plus_pct=("gen_num", lambda x: (x >= 4).mean() * 100),
        gen5_pct=("gen_num", lambda x: (x == 5).mean() * 100),
    ).reset_index()
    result["stage_adoption_curve"] = stage_gen.round(2).to_dict("records")

    # Innovation depth by generation (is there diminishing returns?)
    gen_innov = df.groupby("gen_num")["创新深度评分"].agg(["mean", "std", "count"]).reset_index()
    gen_innov.columns = ["gen_num", "mean_innovation", "std_innovation", "count"]
    result["innovation_by_generation"] = gen_innov.round(3).to_dict("records")

    # Kruskal-Wallis test: innovation across generations
    gen_groups = [grp["创新深度评分"].dropna().values for _, grp in df.groupby("gen_num")]
    gen_groups = [g for g in gen_groups if len(g) >= 5]
    if len(gen_groups) >= 3:
        h_stat, h_p = stats.kruskal(*gen_groups)
        result["generation_innovation_kruskal"] = {
            "H_statistic": round(h_stat, 2),
            "p_value": round(h_p, 6),
            "interpretation": "技术代际间创新深度差异" + ("显著" if h_p < 0.05 else "不显著"),
        }

    return result


# ============================================================
# 7. Policy Implications with Data Support
# ============================================================

def analyze_policy_implications(df):
    """Generate data-backed policy insights."""
    result = {}

    # 7a. 均衡发展指数 across stages
    stage_innov = df[df["学段"].isin(["幼儿园", "小学", "初中", "高中"])].groupby("学段")["创新深度评分"].mean()
    stage_gini = gini_coefficient(stage_innov.values)
    result["stage_balance"] = {
        "gini": round(stage_gini, 4),
        "stage_means": stage_innov.round(2).to_dict(),
        "max_gap": round(stage_innov.max() - stage_innov.min(), 2),
    }

    # 7b. Subject balance
    subj_counts = df["学科"].value_counts()
    valid_subj = subj_counts[subj_counts >= 15].index
    subj_innov = df[df["学科"].isin(valid_subj)].groupby("学科")["创新深度评分"].mean()
    subj_gini = gini_coefficient(subj_innov.values)
    result["subject_balance"] = {
        "gini": round(subj_gini, 4),
        "max_gap": round(subj_innov.max() - subj_innov.min(), 2),
        "highest": f"{subj_innov.idxmax()} ({subj_innov.max():.2f})",
        "lowest": f"{subj_innov.idxmin()} ({subj_innov.min():.2f})",
    }

    # 7c. Regional balance
    prov_innov = df.groupby("省份")["创新深度评分"].mean()
    prov_innov = prov_innov[prov_innov.index != "未提及"]
    prov_gini = gini_coefficient(prov_innov.values)
    result["regional_balance"] = {
        "gini": round(prov_gini, 4),
        "n_provinces": len(prov_innov),
        "max_gap": round(prov_innov.max() - prov_innov.min(), 2),
        "top3": prov_innov.sort_values(ascending=False).head(3).round(2).to_dict(),
        "bottom3": prov_innov.sort_values().head(3).round(2).to_dict(),
    }

    # 7d. 五育失衡 quantification
    wuyu_main = df[df["五育"].isin(["智育", "德育", "美育", "体育", "劳育"])]
    wuyu_dist = wuyu_main["五育"].value_counts()
    total_wuyu = wuyu_dist.sum()
    wuyu_pcts = (wuyu_dist / total_wuyu * 100).round(1)
    result["wuyu_imbalance"] = {
        "distribution_pct": wuyu_pcts.to_dict(),
        "zhiyu_dominance_pct": round(wuyu_pcts.get("智育", 0), 1),
        "non_zhiyu_total_pct": round(100 - wuyu_pcts.get("智育", 0), 1),
        "gini": round(gini_coefficient(wuyu_dist.values), 4),
        "ideal_pct": 20.0,
        "max_deviation_from_ideal": round(abs(wuyu_pcts.max() - 20.0), 1),
    }

    # 7e. HMC(1) to HM2C transition analysis
    hmc1 = df[df["istar_num"] == 1]
    hm2c = df[df["istar_num"] == 2]
    result["hmc_to_hm2c_transition"] = {
        "hmc1_count": len(hmc1),
        "hm2c_count": len(hm2c),
        "hmc1_pct": round(len(hmc1) / len(df) * 100, 1),
        "hm2c_pct": round(len(hm2c) / len(df) * 100, 1),
        "hm2c_gen4plus_pct": round((hm2c["gen_num"] >= 4).mean() * 100, 1),
        "hmc1_gen4plus_pct": round((hmc1["gen_num"] >= 4).mean() * 100, 1),
        "gap_analysis": "HM2C案例中Gen4+工具占比 vs HMC(1)案例",
    }

    # 7f. Biggest policy gaps
    gaps = []
    # Gap 1: 五育
    if wuyu_pcts.get("智育", 0) > 60:
        gaps.append({
            "gap": "五育严重失衡",
            "metric": f"智育占{wuyu_pcts.get('智育', 0):.1f}%, 劳育仅占{wuyu_pcts.get('劳育', 0):.1f}%",
            "severity": "高",
        })
    # Gap 2: Regional
    if prov_gini > 0.1:
        gaps.append({
            "gap": "区域数字鸿沟",
            "metric": f"省份创新深度Gini={prov_gini:.4f}, 最大差距={prov_innov.max() - prov_innov.min():.2f}分",
            "severity": "高" if prov_gini > 0.2 else "中",
        })
    # Gap 3: Stage
    if stage_innov.max() - stage_innov.min() > 0.3:
        gaps.append({
            "gap": "学段发展不均",
            "metric": f"学段间创新深度最大差距={stage_innov.max() - stage_innov.min():.2f}分",
            "severity": "中",
        })
    result["policy_gaps"] = gaps

    return result


# ============================================================
# 8. Industry Ecosystem Health
# ============================================================

def analyze_industry_health(df):
    """Assess the health of the AI education industry ecosystem."""
    result = {}

    # 8a. Market concentration (HHI) by scenario
    scenario_tool_counts = df.groupby("应用场景（一级）")["工具标准名"].value_counts()
    scenario_hhi = {}
    for scenario in df["应用场景（一级）"].dropna().unique():
        if scenario in scenario_tool_counts.index.get_level_values(0):
            counts = scenario_tool_counts[scenario].values
            if len(counts) >= 2:
                scenario_hhi[scenario] = round(hhi_index(counts), 1)
    result["scenario_hhi"] = scenario_hhi
    result["hhi_interpretation"] = "HHI<1500=竞争性, 1500-2500=中度集中, >2500=高度集中"

    # Overall market HHI
    overall_tool_counts = df["工具标准名"].value_counts().values
    result["overall_hhi"] = round(hhi_index(overall_tool_counts), 1)

    # 8b. Long tail effect
    tool_freq = df["工具标准名"].value_counts()
    single_use = (tool_freq == 1).sum()
    result["long_tail"] = {
        "total_unique_tools": len(tool_freq),
        "single_use_tools": int(single_use),
        "single_use_pct": round(single_use / len(tool_freq) * 100, 1),
        "top10_tools_share_pct": round(tool_freq.head(10).sum() / tool_freq.sum() * 100, 1),
        "top3_tools_share_pct": round(tool_freq.head(3).sum() / tool_freq.sum() * 100, 1),
    }

    # 8c. Platform dependency
    company_counts = df["公司"].value_counts()
    top_company = company_counts.index[0] if len(company_counts) > 0 else "N/A"
    top_company_share = round(company_counts.iloc[0] / company_counts.sum() * 100, 1) if len(company_counts) > 0 else 0

    # Bytedance ecosystem (豆包 + 即梦AI + 剪映AI)
    bytedance_tools = ["豆包", "即梦AI", "即梦 AI", "剪映AI", "剪映"]
    bytedance_count = df[df["工具标准名"].isin(bytedance_tools)].shape[0]
    bytedance_pct = round(bytedance_count / len(df) * 100, 1)

    result["platform_dependency"] = {
        "top_company": top_company,
        "top_company_share_pct": top_company_share,
        "bytedance_ecosystem_pct": bytedance_pct,
        "bytedance_ecosystem_count": bytedance_count,
        "top5_companies": company_counts.head(5).to_dict(),
        "company_hhi": round(hhi_index(company_counts.values), 1),
    }

    # 8d. Open-source vs proprietary patterns
    # DeepSeek is open-source; most others are proprietary
    open_source_tools = ["DeepSeek", "DeepSeek 大模型", "Llama", "ChatGLM", "通义千问大模型"]
    os_count = df[df["工具标准名"].isin(open_source_tools)].shape[0]
    result["open_source"] = {
        "open_source_count": os_count,
        "open_source_pct": round(os_count / len(df) * 100, 1),
        "proprietary_pct": round((len(df) - os_count) / len(df) * 100, 1),
        "note": "开源工具主要为DeepSeek系列",
    }

    # 8e. Product form diversity
    product_form = df["产品形态"].value_counts()
    result["product_form_diversity"] = {
        "distribution": product_form.head(10).to_dict(),
        "hhi": round(hhi_index(product_form.values), 1),
    }

    return result


# ============================================================
# Golden Insights Generator
# ============================================================

def generate_golden_insights(results):
    """Extract the top 10 most surprising/actionable findings."""
    insights = []

    # From counter-intuitive
    ci = results.get("counter_intuitive_findings", [])
    for f in ci:
        if f["type"] == "province_econ_vs_innovation":
            insights.append(f"[反直觉] {f['finding']} — {f['interpretation']}")
        elif f["type"] == "low_tech_high_innovation":
            insights.append(f"[反直觉] {f['finding']} (Gen1-2均分={f['gen12_mean']}, Gen4-5均分={f['gen45_mean']}, p={f['p_value']})")
        elif f["type"] == "stage_tech_generation":
            insights.append(f"[学段差异] {f['finding']}")

    # From digital divide
    dd = results.get("digital_divide", {})
    if dd.get("leapfrog_provinces"):
        lf = dd["leapfrog_provinces"][0]
        insights.append(
            f"[跨越式发展] {lf['province']}经济tier={lf.get('econ_tier')}, "
            f"但数字鸿沟指数={lf['digital_divide_index']} (Gen4+占{lf['gen4plus_pct']}%)"
        )
    if dd.get("tool_diversity_gini"):
        insights.append(f"[数字鸿沟] 工具多样性Gini={dd['tool_diversity_gini']} — {dd['tool_diversity_gini_interpretation']}")

    # From tool ecosystem
    te = results.get("tool_ecosystem", {})
    if te.get("multi_vs_single_tool"):
        m = te["multi_vs_single_tool"]
        insights.append(
            f"[工具组合] 多工具案例创新均分={m['multi_tool_mean_innovation']} vs "
            f"单工具={m['single_tool_mean_innovation']} (p={m['p_value']})"
        )

    # From teacher innovation
    ti = results.get("teacher_innovation", {})
    sd = ti.get("self_developed", {})
    if sd.get("mean_innovation") and sd.get("non_self_mean_innovation"):
        insights.append(
            f"[自研工具] 自研工具创新均分={sd['mean_innovation']} vs "
            f"非自研={sd['non_self_mean_innovation']} (p={sd.get('mann_whitney_p', 'N/A')})"
        )

    # From policy
    pi = results.get("policy_implications", {})
    wu = pi.get("wuyu_imbalance", {})
    if wu.get("zhiyu_dominance_pct"):
        insights.append(
            f"[五育失衡] 智育占{wu['zhiyu_dominance_pct']}%, "
            f"非智育仅{wu['non_zhiyu_total_pct']}% (Gini={wu['gini']})"
        )

    # From industry health
    ih = results.get("industry_health", {})
    lt = ih.get("long_tail", {})
    if lt.get("single_use_pct"):
        insights.append(
            f"[长尾效应] {lt['single_use_pct']}%的工具仅被使用1次, "
            f"Top3工具占{lt['top3_tools_share_pct']}%市场份额"
        )
    pd_info = ih.get("platform_dependency", {})
    if pd_info.get("bytedance_ecosystem_pct"):
        insights.append(
            f"[平台依赖] 字节跳动生态(豆包+即梦+剪映)占{pd_info['bytedance_ecosystem_pct']}%"
        )

    # From temporal
    tmp = results.get("temporal_inference", {})
    tp = tmp.get("llm_tipping_point", {})
    if tp.get("llm_era_pct"):
        insights.append(
            f"[技术拐点] LLM时代工具(Gen4+5)占{tp['llm_era_pct']}%, {tp['interpretation']}"
        )

    # Trim to top 10
    return insights[:10]


# PLACEHOLDER_GOLDEN


# PLACEHOLDER_SECTION8


# PLACEHOLDER_SECTION7


# PLACEHOLDER_SECTION6


# PLACEHOLDER_SECTION5


# PLACEHOLDER_SECTION4


def load_data():
    """Load and prepare the V6 CSV data."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    # Normalize key columns
    df["创新深度评分"] = pd.to_numeric(df["创新深度评分"], errors="coerce")
    df["D1_深度学习"] = pd.to_numeric(df["D1_深度学习"], errors="coerce")
    df["D3_循证教学"] = pd.to_numeric(df["D3_循证教学"], errors="coerce")
    df["D4_人机互信"] = pd.to_numeric(df["D4_人机互信"], errors="coerce")
    # Normalize 五育
    df["五育"] = df["培养方向"].apply(normalize_wuyu)
    # Gen numeric
    gen_order = {"Gen1_传统信息化": 1, "Gen2_互联网+": 2, "Gen3_AI辅助": 3,
                 "Gen4_大模型": 4, "Gen5_多模态AI": 5}
    df["gen_num"] = df["产品技术代际"].map(gen_order)
    # iSTAR numeric
    istar_order = {"HUM(0)": 0, "HMC(1)": 1, "HM2C(2)": 2}
    df["istar_num"] = df["iSTAR人机协同层级"].map(istar_order)
    return df


def generate_summary_md(results, golden):
    """Generate a readable markdown summary."""
    lines = [
        "# Deep Insight Mining: 教育AI产业深度洞察报告",
        "",
        "## 核心发现 (Golden Insights)",
        "",
    ]
    for i, g in enumerate(golden, 1):
        lines.append(f"{i}. {g}")
    lines.append("")

    # Counter-intuitive
    lines.append("## 1. 反直觉发现")
    lines.append("")
    ci = results.get("counter_intuitive_findings", [])
    for f in ci:
        lines.append(f"- **{f['type']}**: {f['finding']}")
    lines.append("")

    # Digital divide
    lines.append("## 2. 数字鸿沟深度分析")
    lines.append("")
    dd = results.get("digital_divide", {})
    if dd.get("tool_diversity_gini"):
        lines.append(f"- 工具多样性Gini系数: {dd['tool_diversity_gini']}")
    if dd.get("innovation_gini"):
        lines.append(f"- 创新深度Gini系数: {dd['innovation_gini']}")
    lf = dd.get("leapfrog_provinces", [])
    if lf:
        lines.append(f"- 跨越式发展省份: {', '.join(p['province'] for p in lf[:5])}")
    sd = dd.get("stage_divide", {})
    if sd:
        for stage, info in sd.items():
            lines.append(f"  - {stage}: Gen4+占{info['gen4plus_pct']}%, HM2C占{info['hm2c_pct']}%")
    lines.append("")

    # Tool ecosystem
    lines.append("## 3. 工具生态网络")
    lines.append("")
    te = results.get("tool_ecosystem", {})
    tp = te.get("top_tool_pairs", [])[:10]
    if tp:
        lines.append("### 最常共现工具对")
        lines.append("| 工具1 | 工具2 | 共现次数 |")
        lines.append("|-------|-------|---------|")
        for p in tp:
            lines.append(f"| {p['tool1']} | {p['tool2']} | {p['co_occurrences']} |")
    mvs = te.get("multi_vs_single_tool", {})
    if mvs:
        lines.append(f"\n- 多工具案例创新均分: {mvs['multi_tool_mean_innovation']}")
        lines.append(f"- 单工具案例创新均分: {mvs['single_tool_mean_innovation']}")
        lines.append(f"- Mann-Whitney p={mvs['p_value']}")
    lines.append("")

    return lines


def generate_summary_md_part2(results, lines):
    """Continue generating markdown summary."""
    # Teacher innovation
    lines.append("## 4. 教师创新行为模式")
    lines.append("")
    ti = results.get("teacher_innovation", {})
    sd = ti.get("self_developed", {})
    if sd:
        lines.append(f"- 自研工具占比: {sd.get('pct_of_total', 'N/A')}%")
        lines.append(f"- 自研创新均分: {sd.get('mean_innovation', 'N/A')} vs 非自研: {sd.get('non_self_mean_innovation', 'N/A')}")
    pi = ti.get("pioneers_top5pct", {})
    if pi:
        lines.append(f"- 先行者(Top 5%): {pi.get('count', 0)}个案例, Gen均值={pi.get('mean_gen', 'N/A')}, HM2C占{pi.get('hm2c_pct', 'N/A')}%")
    lines.append("")

    # Subject-AI fit
    lines.append("## 5. 学科-AI适配度矩阵")
    lines.append("")
    saf = results.get("subject_ai_fit", {})
    ar = saf.get("ai_ready_subjects", [])
    ars = saf.get("ai_resistant_subjects", [])
    if ar:
        lines.append(f"- AI-ready学科: {', '.join(ar)}")
    if ars:
        lines.append(f"- AI-resistant学科: {', '.join(ars)}")
    wu = saf.get("wuyu_coverage", [])
    if wu:
        lines.append("\n### 五育AI覆盖")
        lines.append("| 五育 | 案例数 | 创新均分 | HM2C占比 |")
        lines.append("|------|--------|---------|---------|")
        for w in wu[:6]:
            lines.append(f"| {w.get('五育', '')} | {w.get('count', '')} | {w.get('mean_innovation', '')} | {w.get('hm2c_pct', '')}% |")
    lines.append("")

    # Temporal
    lines.append("## 6. 时间序列推断")
    lines.append("")
    tmp = results.get("temporal_inference", {})
    tp = tmp.get("llm_tipping_point", {})
    if tp:
        lines.append(f"- LLM时代工具占比: {tp.get('llm_era_pct', 'N/A')}%")
        lines.append(f"- {tp.get('interpretation', '')}")
    lines.append("")

    # Policy
    lines.append("## 7. 政策启示")
    lines.append("")
    pol = results.get("policy_implications", {})
    wu_imb = pol.get("wuyu_imbalance", {})
    if wu_imb:
        lines.append(f"- 五育失衡: 智育占{wu_imb.get('zhiyu_dominance_pct', 'N/A')}%")
    gaps = pol.get("policy_gaps", [])
    for g in gaps:
        lines.append(f"- **{g['gap']}** [{g['severity']}]: {g['metric']}")
    lines.append("")

    # Industry health
    lines.append("## 8. 产业生态健康度")
    lines.append("")
    ih = results.get("industry_health", {})
    lt = ih.get("long_tail", {})
    if lt:
        lines.append(f"- 工具总数: {lt.get('total_unique_tools', 'N/A')}")
        lines.append(f"- 单次使用工具: {lt.get('single_use_pct', 'N/A')}%")
        lines.append(f"- Top3工具市场份额: {lt.get('top3_tools_share_pct', 'N/A')}%")
    lines.append(f"- 整体HHI: {ih.get('overall_hhi', 'N/A')}")
    pd_info = ih.get("platform_dependency", {})
    if pd_info:
        lines.append(f"- 字节跳动生态占比: {pd_info.get('bytedance_ecosystem_pct', 'N/A')}%")
    lines.append("")

    return "\n".join(lines)


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def main():
    """Run all analyses and save results."""
    print("=" * 60)
    print("Deep Insight Mining: 从数据中提取真金")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows, {df['案例编号'].nunique()} unique cases")

    results = {}

    print("\n[1/8] Analyzing counter-intuitive findings...")
    results["counter_intuitive_findings"] = analyze_counter_intuitive(df)

    print("[2/8] Analyzing digital divide...")
    results["digital_divide"] = analyze_digital_divide(df)

    print("[3/8] Analyzing tool ecosystem...")
    results["tool_ecosystem"] = analyze_tool_ecosystem(df)

    print("[4/8] Analyzing teacher innovation patterns...")
    results["teacher_innovation"] = analyze_teacher_innovation(df)

    print("[5/8] Building subject-AI fit matrix...")
    results["subject_ai_fit"] = analyze_subject_ai_fit(df)

    print("[6/8] Temporal inference...")
    results["temporal_inference"] = analyze_temporal_inference(df)

    print("[7/8] Policy implications...")
    results["policy_implications"] = analyze_policy_implications(df)

    print("[8/8] Industry ecosystem health...")
    results["industry_health"] = analyze_industry_health(df)

    print("\nGenerating golden insights...")
    golden = generate_golden_insights(results)
    results["golden_insights"] = golden

    # Save JSON
    json_path = OUTPUT / "deep_insights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {json_path}")

    # Save markdown summary
    md_lines = generate_summary_md(results, golden)
    md_text = generate_summary_md_part2(results, md_lines)
    md_path = OUTPUT / "deep_insights_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved: {md_path}")

    # Print golden insights
    print("\n" + "=" * 60)
    print("TOP 10 GOLDEN INSIGHTS")
    print("=" * 60)
    for i, g in enumerate(golden, 1):
        print(f"\n  {i}. {g}")
    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    main()
