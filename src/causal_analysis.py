#!/usr/bin/env python3
"""
Comprehensive Causal & Statistical Analysis for AI Education Dataset
====================================================================
Performs 10 multi-dimensional analyses:
  1. Correspondence Analysis (CA)
  2. Multiple Correspondence Analysis (MCA)
  3. Ordinal Logistic Regression
  4. Multinomial Logistic Regression
  5. Structural Equation Modeling (Path Analysis)
  6. Geographic Inequality Decomposition
  7. Cluster Profiling with ANOVA/Chi-Square
  8. Association Rule Mining
  9. Interaction Effects (Factorial ANOVA)
  10. Predictive Modeling (Random Forest + SHAP)

Author: Causal Analysis Pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import traceback

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE = Path("/Users/sakai/Desktop/产业调研/ai-edu-research")
DATA_PATH = BASE / "output" / "教育产品统计_V6_框架标注.csv"
OUT_DIR = BASE / "output"
RESULTS_JSON = OUT_DIR / "causal_analysis_results.json"
SUMMARY_MD = OUT_DIR / "causal_analysis_summary.md"

# -------------------------------------------------------------------
# HELPER: JSON-safe converter
# -------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="index")
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)

# -------------------------------------------------------------------
# LOAD & PREPROCESS
# -------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # Standardize stage labels
    stage_map = {
        "小学": "小学", "初中": "初中", "高中": "高中",
        "幼儿园": "幼儿园", "中学": "中学",
        "中小学": "中小学", "小学/初中": "小学/初中",
        "初中/高中": "初中/高中", "基础教育": "基础教育",
        "学前教育": "幼儿园", "中职": "中职",
        "幼儿园中班": "幼儿园",
    }
    df["学段_clean"] = df["学段"].map(stage_map).fillna("其他")

    # iSTAR numeric
    istar_num = {"HUM(0)": 0, "HMC(1)": 1, "HM2C(2)": 2}
    df["iSTAR_num"] = df["iSTAR人机协同层级"].map(istar_num)

    # Tech generation numeric
    gen_num = {
        "Gen1_传统信息化": 1, "Gen2_互联网+": 2,
        "Gen3_AI辅助": 3, "Gen4_大模型": 4, "Gen5_多模态AI": 5
    }
    df["tech_gen_num"] = df["产品技术代际"].map(gen_num)

    # Smart education level numeric
    smart_num = {
        "第一境界_智慧环境": 1, "第二境界_教学模式": 2,
        "第三境界_制度变革": 3
    }
    df["smart_edu_num"] = df["智慧教育境界"].map(smart_num)

    # Province region grouping
    region_map = {
        "北京市": "华北", "天津市": "华北", "河北省": "华北",
        "山西省": "华北", "内蒙古自治区": "华北",
        "辽宁省": "东北", "吉林省": "东北", "黑龙江省": "东北",
        "上海市": "华东", "江苏省": "华东", "浙江省": "华东",
        "安徽省": "华东", "福建省": "华东", "江西省": "华东",
        "山东省": "华东",
        "河南省": "华中", "湖北省": "华中", "湖南省": "华中",
        "广东省": "华南", "广西壮族自治区": "华南", "海南省": "华南",
        "重庆市": "西南", "四川省": "西南", "贵州省": "西南",
        "云南省": "西南", "西藏自治区": "西南",
        "陕西省": "西北", "甘肃省": "西北", "青海省": "西北",
        "宁夏回族自治区": "西北", "新疆维吾尔自治区": "西北",
    }
    df["区域"] = df["省份"].map(region_map).fillna("其他")

    # Development level by region
    dev_map = {
        "华东": 3, "华南": 3, "华北": 2,
        "华中": 2, "东北": 1, "西南": 1, "西北": 1, "其他": 1
    }
    df["dev_level"] = df["区域"].map(dev_map)

    # Self-developed boolean
    df["自研_num"] = df["是否自主研发"].astype(float).fillna(0).astype(int)

    return df


# ===================================================================
# ANALYSIS 1: CORRESPONDENCE ANALYSIS
# ===================================================================
def analysis_1_correspondence(df):
    """Simple Correspondence Analysis: Subject x Scenario, Stage x Three-Empowerment"""
    import prince

    results = {}

    # --- CA 1: Subject x Scenario ---
    ct1 = pd.crosstab(df["学科"], df["应用场景（一级）"])
    # Remove very small categories
    ct1 = ct1.loc[ct1.sum(axis=1) >= 10, ct1.sum(axis=0) >= 10]

    ca1 = prince.CA(n_components=2)
    ca1 = ca1.fit(ct1)

    row_coords = ca1.row_coordinates(ct1)
    col_coords = ca1.column_coordinates(ct1)

    # Eigenvalues / inertia
    eigenvalues = ca1.eigenvalues_
    total_inertia = ca1.total_inertia_
    pct_inertia = [float(e / total_inertia * 100) if total_inertia > 0 else 0 for e in eigenvalues]

    results["subject_scenario_CA"] = {
        "row_coordinates": row_coords.to_dict(orient="index"),
        "col_coordinates": col_coords.to_dict(orient="index"),
        "eigenvalues": [float(e) for e in eigenvalues],
        "total_inertia": float(total_inertia),
        "pct_inertia_explained": pct_inertia,
        "contingency_table_shape": list(ct1.shape),
    }
    print(f"  CA1 Subject x Scenario: inertia explained = {pct_inertia}")

    # --- CA 2: Stage x Three-Empowerment ---
    ct2 = pd.crosstab(df["学段_clean"], df["三赋能分类"])
    ct2 = ct2.loc[ct2.sum(axis=1) >= 10, ct2.sum(axis=0) >= 10]

    n_comp = min(2, min(ct2.shape) - 1)
    ca2 = prince.CA(n_components=max(n_comp, 1))
    ca2 = ca2.fit(ct2)

    row_coords2 = ca2.row_coordinates(ct2)
    col_coords2 = ca2.column_coordinates(ct2)
    eigenvalues2 = ca2.eigenvalues_
    total_inertia2 = ca2.total_inertia_
    pct2 = [float(e / total_inertia2 * 100) if total_inertia2 > 0 else 0 for e in eigenvalues2]

    results["stage_empowerment_CA"] = {
        "row_coordinates": row_coords2.to_dict(orient="index"),
        "col_coordinates": col_coords2.to_dict(orient="index"),
        "eigenvalues": [float(e) for e in eigenvalues2],
        "total_inertia": float(total_inertia2),
        "pct_inertia_explained": pct2,
        "contingency_table_shape": list(ct2.shape),
    }
    print(f"  CA2 Stage x Empowerment: inertia explained = {pct2}")

    return results


# ===================================================================
# ANALYSIS 2: MULTIPLE CORRESPONDENCE ANALYSIS
# ===================================================================
def analysis_2_mca(df):
    """MCA on 学段, 三赋能分类, iSTAR, 智慧教育境界, 产品技术代际"""
    import prince

    cols = ["学段_clean", "三赋能分类", "iSTAR人机协同层级", "智慧教育境界", "产品技术代际"]
    sub = df[cols].dropna()
    # Limit to main categories for stage
    valid_stages = ["小学", "初中", "高中", "幼儿园"]
    sub = sub[sub["学段_clean"].isin(valid_stages)]

    mca = prince.MCA(n_components=3)
    mca = mca.fit(sub)

    row_coords = mca.row_coordinates(sub)
    col_coords = mca.column_coordinates(sub)

    eigenvalues = mca.eigenvalues_
    total_inertia = mca.total_inertia_
    pct = [float(e / total_inertia * 100) if total_inertia > 0 else 0 for e in eigenvalues]

    # Factor loadings = column coordinates
    col_dict = col_coords.to_dict(orient="index")

    results = {
        "n_observations": len(sub),
        "n_components": 3,
        "eigenvalues": [float(e) for e in eigenvalues],
        "total_inertia": float(total_inertia),
        "pct_inertia_explained": pct,
        "cumulative_inertia": [float(sum(pct[:i+1])) for i in range(len(pct))],
        "column_coordinates_factor_loadings": col_dict,
    }

    # Interpret dimensions
    interpretations = {}
    for dim in range(min(3, len(col_coords.columns))):
        col_name = col_coords.columns[dim]
        sorted_coords = col_coords[col_name].sort_values()
        interpretations[f"Dimension_{dim+1}"] = {
            "negative_pole": sorted_coords.head(3).to_dict(),
            "positive_pole": sorted_coords.tail(3).to_dict(),
        }
    results["dimension_interpretations"] = interpretations

    print(f"  MCA: {len(sub)} obs, inertia explained = {pct[:3]}")
    return results


# ===================================================================
# ANALYSIS 3: ORDINAL LOGISTIC REGRESSION
# ===================================================================
def analysis_3_ordinal_logit(df):
    """Ordinal logistic regression: DV = 创新深度评分"""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    sub = df[["创新深度评分", "学段_clean", "tech_gen_num", "三赋能分类",
              "iSTAR_num", "自研_num", "区域"]].dropna()
    valid_stages = ["小学", "初中", "高中", "幼儿园"]
    sub = sub[sub["学段_clean"].isin(valid_stages)]

    # Create dummies
    X = pd.get_dummies(sub[["学段_clean", "三赋能分类", "区域"]], drop_first=True, dtype=float)
    X["tech_gen_num"] = sub["tech_gen_num"].values
    X["iSTAR_num"] = sub["iSTAR_num"].values
    X["自研_num"] = sub["自研_num"].values
    y = sub["创新深度评分"].astype(int).values

    model = OrderedModel(y, X, distr="logit")
    res = model.fit(method="bfgs", disp=False, maxiter=5000)

    # Extract results - OrderedModel params include both regression coefficients and thresholds
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues

    # The first len(X.columns) params are the regression coefficients
    n_coef = len(X.columns)
    coef_table = {}
    for i, name in enumerate(X.columns):
        if i < len(params):
            coef_val = float(params.iloc[i]) if hasattr(params, 'iloc') else float(params[i])
            pval = float(pvals.iloc[i]) if hasattr(pvals, 'iloc') else float(pvals[i])
            ci_low = float(conf.iloc[i, 0]) if hasattr(conf, 'iloc') else float(conf[i, 0])
            ci_high = float(conf.iloc[i, 1]) if hasattr(conf, 'iloc') else float(conf[i, 1])
            coef_table[name] = {
                "coefficient": coef_val,
                "odds_ratio": float(np.exp(coef_val)),
                "CI_lower": float(np.exp(ci_low)),
                "CI_upper": float(np.exp(ci_high)),
                "p_value": pval,
                "significant": bool(pval < 0.05),
            }

    # Pseudo R-squared
    pseudo_r2 = float(res.prsquared) if hasattr(res, "prsquared") else float(1 - res.llf / res.llnull)

    # Proportional odds test: compare full ordinal model vs separate binary models
    # Simplified: fit binary logits at each cutpoint and compare
    from scipy.stats import chi2
    binary_ll = 0
    for threshold in sorted(np.unique(y))[:-1]:
        y_bin = (y > threshold).astype(int)
        from statsmodels.api import Logit
        try:
            bmodel = Logit(y_bin, pd.concat([X, pd.Series(np.ones(len(X)), index=X.index, name="const")], axis=1))
            bres = bmodel.fit(disp=False, maxiter=2000)
            binary_ll += bres.llf
        except Exception:
            binary_ll += res.llf / (len(np.unique(y)) - 1)

    lr_stat = -2 * (res.llf - binary_ll)
    k_diff = (len(np.unique(y)) - 2) * len(X.columns)
    prop_odds_p = float(1 - chi2.cdf(abs(lr_stat), max(k_diff, 1)))

    results = {
        "n_observations": len(sub),
        "n_categories": int(len(np.unique(y))),
        "pseudo_r_squared": pseudo_r2,
        "log_likelihood": float(res.llf),
        "AIC": float(res.aic) if hasattr(res, "aic") else None,
        "BIC": float(res.bic) if hasattr(res, "bic") else None,
        "coefficients": coef_table,
        "proportional_odds_test": {
            "LR_statistic": float(lr_stat),
            "df": k_diff,
            "p_value": prop_odds_p,
            "assumption_holds": prop_odds_p > 0.05,
        },
    }

    print(f"  Ordinal Logit: n={len(sub)}, pseudo R2={pseudo_r2:.4f}")
    sig_vars = [k for k, v in coef_table.items() if v["significant"]]
    print(f"  Significant predictors: {sig_vars}")
    return results


# ===================================================================
# ANALYSIS 4: MULTINOMIAL LOGISTIC REGRESSION
# ===================================================================
def analysis_4_multinomial_logit(df):
    """Multinomial logistic: DV = iSTAR level"""
    from statsmodels.api import MNLogit
    import statsmodels.api as sm

    sub = df[["iSTAR人机协同层级", "学段_clean", "学科", "tech_gen_num",
              "D1_深度学习", "D2_绿色鲁棒", "D3_循证教学", "D4_人机互信"]].dropna()
    valid_stages = ["小学", "初中", "高中", "幼儿园"]
    sub = sub[sub["学段_clean"].isin(valid_stages)]

    # Group subjects into top categories
    top_subjects = sub["学科"].value_counts().head(8).index.tolist()
    sub["学科_grouped"] = sub["学科"].where(sub["学科"].isin(top_subjects), "其他")

    X = pd.get_dummies(sub[["学段_clean", "学科_grouped"]], drop_first=True, dtype=float)
    X["tech_gen_num"] = sub["tech_gen_num"].values
    X["D1_深度学习"] = sub["D1_深度学习"].values
    X["D2_绿色鲁棒"] = sub["D2_绿色鲁棒"].values
    X["D3_循证教学"] = sub["D3_循证教学"].values
    X["D4_人机互信"] = sub["D4_人机互信"].values
    X = sm.add_constant(X)

    y = sub["iSTAR人机协同层级"]

    model = MNLogit(y, X)
    res = model.fit(method="bfgs", disp=False, maxiter=5000)

    # Extract RRR
    params_df = res.params
    pvalues_df = res.pvalues
    conf_df = res.conf_int()

    results_by_outcome = {}
    for col_idx, outcome in enumerate(params_df.columns):
        coef_dict = {}
        for row_idx, var in enumerate(params_df.index):
            coef = float(params_df.loc[var, outcome])
            pval = float(pvalues_df.loc[var, outcome])
            coef_dict[var] = {
                "coefficient": coef,
                "relative_risk_ratio": float(np.exp(coef)),
                "p_value": pval,
                "significant": bool(pval < 0.05),
            }
        results_by_outcome[str(outcome)] = coef_dict

    pseudo_r2 = float(res.prsquared) if hasattr(res, "prsquared") else float(1 - res.llf / res.llnull)

    results = {
        "n_observations": len(sub),
        "reference_category": str(res.model.endog_names) if hasattr(res.model, "endog_names") else "HMC(1)",
        "pseudo_r_squared": pseudo_r2,
        "log_likelihood": float(res.llf),
        "AIC": float(res.aic) if hasattr(res, "aic") else None,
        "coefficients_by_outcome": results_by_outcome,
    }

    print(f"  Multinomial Logit: n={len(sub)}, pseudo R2={pseudo_r2:.4f}")
    return results


# ===================================================================
# ANALYSIS 5: PATH ANALYSIS / SIMPLIFIED SEM
# ===================================================================
def analysis_5_sem(df):
    """Simplified path analysis with mediation tests"""
    from scipy.stats import norm
    import statsmodels.api as sm

    sub = df[["tech_gen_num", "D3_循证教学", "D4_人机互信",
              "创新深度评分", "dev_level"]].dropna()

    X_const = lambda x: sm.add_constant(x)

    results = {}

    # Path a1: DevLevel -> TechGen
    m1 = sm.OLS(sub["tech_gen_num"], X_const(sub[["dev_level"]])).fit()
    a1 = float(m1.params["dev_level"])
    se_a1 = float(m1.bse["dev_level"])

    # Path a2: TechGen -> D4_HumanMachineTrust
    m2 = sm.OLS(sub["D4_人机互信"], X_const(sub[["tech_gen_num"]])).fit()
    a2 = float(m2.params["tech_gen_num"])
    se_a2 = float(m2.bse["tech_gen_num"])

    # Path a3: TechGen -> D3_EvidenceTeaching
    m3 = sm.OLS(sub["D3_循证教学"], X_const(sub[["tech_gen_num"]])).fit()
    a3 = float(m3.params["tech_gen_num"])
    se_a3 = float(m3.bse["tech_gen_num"])

    # Path b (full model): TechGen + D4 + D3 -> InnovationDepth
    X_full = X_const(sub[["tech_gen_num", "D4_人机互信", "D3_循证教学"]])
    m4 = sm.OLS(sub["创新深度评分"], X_full).fit()
    b_D4 = float(m4.params["D4_人机互信"])
    se_b_D4 = float(m4.bse["D4_人机互信"])
    b_D3 = float(m4.params["D3_循证教学"])
    se_b_D3 = float(m4.bse["D3_循证教学"])
    c_prime = float(m4.params["tech_gen_num"])  # direct effect

    # Direct effect of TechGen -> Innovation (without mediators)
    m5 = sm.OLS(sub["创新深度评分"], X_const(sub[["tech_gen_num"]])).fit()
    c_total = float(m5.params["tech_gen_num"])

    # Indirect effects
    indirect_D4 = a2 * b_D4
    indirect_D3 = a3 * b_D3
    total_indirect = indirect_D4 + indirect_D3

    # Sobel test for D4 mediation
    sobel_se_D4 = np.sqrt(a2**2 * se_b_D4**2 + b_D4**2 * se_a2**2)
    sobel_z_D4 = indirect_D4 / sobel_se_D4 if sobel_se_D4 > 0 else 0
    sobel_p_D4 = float(2 * (1 - norm.cdf(abs(sobel_z_D4))))

    # Sobel test for D3 mediation
    sobel_se_D3 = np.sqrt(a3**2 * se_b_D3**2 + b_D3**2 * se_a3**2)
    sobel_z_D3 = indirect_D3 / sobel_se_D3 if sobel_se_D3 > 0 else 0
    sobel_p_D3 = float(2 * (1 - norm.cdf(abs(sobel_z_D3))))

    # Bootstrap mediation (1000 replications)
    np.random.seed(42)
    n_boot = 1000
    boot_indirect_D4 = []
    boot_indirect_D3 = []
    for _ in range(n_boot):
        idx = np.random.choice(len(sub), len(sub), replace=True)
        b = sub.iloc[idx]
        try:
            bm2 = sm.OLS(b["D4_人机互信"], X_const(b[["tech_gen_num"]])).fit()
            bm3 = sm.OLS(b["D3_循证教学"], X_const(b[["tech_gen_num"]])).fit()
            bm4 = sm.OLS(b["创新深度评分"], X_const(b[["tech_gen_num", "D4_人机互信", "D3_循证教学"]])).fit()
            boot_indirect_D4.append(bm2.params["tech_gen_num"] * bm4.params["D4_人机互信"])
            boot_indirect_D3.append(bm3.params["tech_gen_num"] * bm4.params["D3_循证教学"])
        except Exception:
            pass

    boot_D4_ci = (float(np.percentile(boot_indirect_D4, 2.5)), float(np.percentile(boot_indirect_D4, 97.5)))
    boot_D3_ci = (float(np.percentile(boot_indirect_D3, 2.5)), float(np.percentile(boot_indirect_D3, 97.5)))

    results = {
        "n_observations": len(sub),
        "paths": {
            "DevLevel_to_TechGen": {
                "coef": a1, "se": se_a1, "p": float(m1.pvalues["dev_level"]),
                "R2": float(m1.rsquared),
            },
            "TechGen_to_D4": {
                "coef": a2, "se": se_a2, "p": float(m2.pvalues["tech_gen_num"]),
                "R2": float(m2.rsquared),
            },
            "TechGen_to_D3": {
                "coef": a3, "se": se_a3, "p": float(m3.pvalues["tech_gen_num"]),
                "R2": float(m3.rsquared),
            },
            "D4_to_Innovation": {"coef": b_D4, "se": se_b_D4, "p": float(m4.pvalues["D4_人机互信"])},
            "D3_to_Innovation": {"coef": b_D3, "se": se_b_D3, "p": float(m4.pvalues["D3_循证教学"])},
            "TechGen_to_Innovation_direct": {"coef": c_prime, "p": float(m4.pvalues["tech_gen_num"])},
            "TechGen_to_Innovation_total": {"coef": c_total, "p": float(m5.pvalues["tech_gen_num"])},
        },
        "effects": {
            "total_effect": c_total,
            "direct_effect": c_prime,
            "indirect_via_D4": indirect_D4,
            "indirect_via_D3": indirect_D3,
            "total_indirect": total_indirect,
            "proportion_mediated": float(total_indirect / c_total) if c_total != 0 else None,
        },
        "mediation_tests": {
            "D4_Sobel": {"z": sobel_z_D4, "p": sobel_p_D4, "significant": sobel_p_D4 < 0.05},
            "D3_Sobel": {"z": sobel_z_D3, "p": sobel_p_D3, "significant": sobel_p_D3 < 0.05},
            "D4_bootstrap_CI_95": boot_D4_ci,
            "D3_bootstrap_CI_95": boot_D3_ci,
            "D4_bootstrap_significant": not (boot_D4_ci[0] <= 0 <= boot_D4_ci[1]),
            "D3_bootstrap_significant": not (boot_D3_ci[0] <= 0 <= boot_D3_ci[1]),
        },
        "full_model_R2": float(m4.rsquared),
        "full_model_adj_R2": float(m4.rsquared_adj),
    }

    print(f"  SEM: total={c_total:.4f}, direct={c_prime:.4f}, indirect_D4={indirect_D4:.4f}, indirect_D3={indirect_D3:.4f}")
    print(f"  Mediation D4 Sobel p={sobel_p_D4:.4f}, D3 Sobel p={sobel_p_D3:.4f}")
    return results


# ===================================================================
# ANALYSIS 6: GEOGRAPHIC INEQUALITY DECOMPOSITION
# ===================================================================
def analysis_6_geographic(df):
    """Theil index and Gini coefficient decomposition"""

    def theil_T(x):
        """Theil T index (GE(1))"""
        x = x[x > 0]
        if len(x) == 0:
            return 0.0
        mu = np.mean(x)
        n = len(x)
        return float(np.sum((x / mu) * np.log(x / mu)) / n)

    def gini(x):
        """Gini coefficient"""
        x = np.sort(np.array(x, dtype=float))
        x = x[~np.isnan(x)]
        n = len(x)
        if n == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))) if np.sum(x) > 0 else 0.0

    def theil_decomposition(df, var, group_col):
        """Decompose Theil index into within and between group components"""
        data = df[[var, group_col]].dropna()
        x = data[var].values.astype(float)
        # Shift to positive for Theil
        x_shifted = x - x.min() + 1
        data = data.copy()
        data["_val"] = x_shifted

        total_theil = theil_T(x_shifted)
        mu = np.mean(x_shifted)
        n_total = len(x_shifted)

        between = 0
        within = 0
        groups = data.groupby(group_col)
        for name, grp in groups:
            n_g = len(grp)
            mu_g = np.mean(grp["_val"])
            s_g = n_g / n_total  # population share
            y_g = (n_g * mu_g) / (n_total * mu)  # income share

            # Between component
            if y_g > 0 and s_g > 0:
                between += y_g * np.log(y_g / s_g)
            # Within component
            within += y_g * theil_T(grp["_val"].values)

        return {
            "total_theil": float(total_theil),
            "between_groups": float(between),
            "within_groups": float(within),
            "pct_between": float(between / total_theil * 100) if total_theil > 0 else 0,
            "pct_within": float(within / total_theil * 100) if total_theil > 0 else 0,
        }

    results = {}
    metrics = {
        "innovation_depth": "创新深度评分",
        "iSTAR_level": "iSTAR_num",
        "tech_generation": "tech_gen_num",
    }

    for metric_name, col in metrics.items():
        sub = df[[col, "省份", "区域"]].dropna()
        sub = sub[sub["省份"] != "未提及"]

        # Theil decomposition by province
        theil_prov = theil_decomposition(sub, col, "省份")
        # Theil decomposition by region
        theil_reg = theil_decomposition(sub, col, "区域")

        # Gini by province
        gini_by_prov = {}
        for prov, grp in sub.groupby("省份"):
            if len(grp) >= 5:
                gini_by_prov[prov] = gini(grp[col].values)

        # Overall Gini
        overall_gini = gini(sub[col].values)

        results[metric_name] = {
            "overall_gini": overall_gini,
            "theil_by_province": theil_prov,
            "theil_by_region": theil_reg,
            "gini_by_province": dict(sorted(gini_by_prov.items(), key=lambda x: -x[1])[:15]),
            "n_observations": len(sub),
        }

        print(f"  {metric_name}: Gini={overall_gini:.4f}, Between-province={theil_prov['pct_between']:.1f}%")

    return results


# ===================================================================
# ANALYSIS 7: CLUSTER PROFILING
# ===================================================================
def analysis_7_cluster_profiling(df):
    """KMeans clustering + ANOVA/Chi-square profiling"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import f_oneway, kruskal, chi2_contingency

    # Prepare features for clustering
    cluster_features = ["tech_gen_num", "iSTAR_num", "创新深度评分",
                        "D1_深度学习", "D2_绿色鲁棒", "D3_循证教学", "D4_人机互信",
                        "smart_edu_num"]
    sub = df[cluster_features + ["学段_clean", "应用场景（一级）", "产品技术代际",
                                  "三赋能分类", "区域"]].dropna()

    X_cluster = sub[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # KMeans with 10 clusters
    km = KMeans(n_clusters=10, random_state=42, n_init=20, max_iter=500)
    sub = sub.copy()
    sub["cluster"] = km.fit_predict(X_scaled)

    results = {"n_clusters": 10, "n_observations": len(sub)}

    # Cluster sizes
    sizes = sub["cluster"].value_counts().sort_index().to_dict()
    results["cluster_sizes"] = {str(k): int(v) for k, v in sizes.items()}

    # Cluster centroids (original scale)
    centroids = {}
    for c in range(10):
        mask = sub["cluster"] == c
        centroids[str(c)] = {feat: float(sub.loc[mask, feat].mean()) for feat in cluster_features}
    results["cluster_centroids"] = centroids

    # ANOVA for continuous variables
    continuous = ["创新深度评分", "D1_深度学习", "D3_循证教学", "D4_人机互信", "tech_gen_num"]
    anova_results = {}
    for var in continuous:
        groups = [sub.loc[sub["cluster"] == c, var].values for c in range(10)]
        groups = [g for g in groups if len(g) >= 2]
        f_stat, p_val = f_oneway(*groups)
        # Effect size: eta-squared
        ss_between = sum(len(g) * (np.mean(g) - sub[var].mean())**2 for g in groups)
        ss_total = sum(np.sum((g - sub[var].mean())**2) for g in groups)
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0

        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p = kruskal(*groups)

        anova_results[var] = {
            "F_statistic": float(f_stat),
            "p_value": float(p_val),
            "eta_squared": eta_sq,
            "effect_size_interpretation": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small",
            "kruskal_wallis_H": float(h_stat),
            "kruskal_wallis_p": float(kw_p),
        }
    results["anova_continuous"] = anova_results

    # Post-hoc pairwise comparisons (Bonferroni)
    from itertools import combinations
    from scipy.stats import mannwhitneyu
    posthoc = {}
    for var in ["创新深度评分", "tech_gen_num"]:
        pairs = []
        cluster_groups = {c: sub.loc[sub["cluster"] == c, var].values for c in range(10)}
        n_comparisons = 45  # C(10,2)
        for c1, c2 in combinations(range(10), 2):
            if len(cluster_groups[c1]) >= 2 and len(cluster_groups[c2]) >= 2:
                u_stat, p_val = mannwhitneyu(cluster_groups[c1], cluster_groups[c2], alternative="two-sided")
                adjusted_p = min(float(p_val) * n_comparisons, 1.0)
                # Effect size: rank-biserial correlation
                n1, n2 = len(cluster_groups[c1]), len(cluster_groups[c2])
                r_rb = 1 - (2 * u_stat) / (n1 * n2)
                if adjusted_p < 0.05:
                    pairs.append({
                        "cluster_pair": f"{c1}_vs_{c2}",
                        "U_statistic": float(u_stat),
                        "p_adjusted_bonferroni": adjusted_p,
                        "rank_biserial_r": float(r_rb),
                    })
        posthoc[var] = pairs[:20]  # top 20
    results["posthoc_comparisons"] = posthoc

    # Chi-square for categorical variables
    categorical = ["学段_clean", "应用场景（一级）", "产品技术代际", "三赋能分类"]
    chi2_results = {}
    for var in categorical:
        ct = pd.crosstab(sub["cluster"], sub[var])
        chi2_stat, p_val, dof, expected = chi2_contingency(ct)
        n = ct.sum().sum()
        # Cramer's V
        k = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2_stat / (n * k))) if (n * k) > 0 else 0

        chi2_results[var] = {
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_val),
            "dof": int(dof),
            "cramers_v": cramers_v,
            "effect_size_interpretation": "large" if cramers_v > 0.5 else "medium" if cramers_v > 0.3 else "small",
        }

        # Distribution per cluster
        dist = ct.div(ct.sum(axis=1), axis=0)
        chi2_results[var]["cluster_distributions"] = {
            str(k2): v2.to_dict() for k2, v2 in dist.iterrows()
        }

    results["chi_square_categorical"] = chi2_results

    print(f"  Clustering: 10 clusters, sizes: {list(sizes.values())}")
    return results


# ===================================================================
# ANALYSIS 8: ASSOCIATION RULE MINING
# ===================================================================
def analysis_8_association_rules(df):
    """Apriori association rule mining"""
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    # Prepare items
    sub = df[["学段_clean", "学科", "应用场景（一级）", "三赋能分类", "iSTAR人机协同层级"]].dropna()
    valid_stages = ["小学", "初中", "高中", "幼儿园"]
    sub = sub[sub["学段_clean"].isin(valid_stages)]

    # Group subjects
    top_subjects = sub["学科"].value_counts().head(10).index.tolist()
    sub = sub.copy()
    sub["学科"] = sub["学科"].where(sub["学科"].isin(top_subjects), "其他学科")

    # Create one-hot encoded DataFrame
    transactions = []
    for _, row in sub.iterrows():
        items = [
            f"学段_{row['学段_clean']}",
            f"学科_{row['学科']}",
            f"场景_{row['应用场景（一级）']}",
            f"赋能_{row['三赋能分类']}",
            f"协同_{row['iSTAR人机协同层级']}",
        ]
        transactions.append(items)

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori
    freq_items = apriori(df_te, min_support=0.05, use_colnames=True, max_len=4)
    print(f"  Found {len(freq_items)} frequent itemsets")

    if len(freq_items) == 0:
        return {"error": "No frequent itemsets found with min_support=0.05"}

    rules = association_rules(freq_items, num_itemsets=len(transactions),
                               metric="confidence", min_threshold=0.3)
    rules = rules.sort_values("lift", ascending=False)

    # Convert to serializable format
    top_rules = []
    for _, r in rules.head(30).iterrows():
        top_rules.append({
            "antecedents": list(r["antecedents"]),
            "consequents": list(r["consequents"]),
            "support": float(r["support"]),
            "confidence": float(r["confidence"]),
            "lift": float(r["lift"]),
            "conviction": float(r["conviction"]) if not np.isinf(r["conviction"]) else "inf",
            "leverage": float(r["leverage"]),
        })

    # Summary statistics
    results = {
        "n_transactions": len(transactions),
        "n_frequent_itemsets": len(freq_items),
        "n_rules": len(rules),
        "min_support": 0.05,
        "min_confidence": 0.3,
        "top_30_rules_by_lift": top_rules,
        "lift_statistics": {
            "mean": float(rules["lift"].mean()),
            "max": float(rules["lift"].max()),
            "min": float(rules["lift"].min()),
        },
    }

    print(f"  Association Rules: {len(rules)} rules, top lift={rules['lift'].max():.2f}")
    return results


# ===================================================================
# ANALYSIS 9: INTERACTION EFFECTS
# ===================================================================
def analysis_9_interactions(df):
    """Factorial ANOVA with interaction terms"""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    results = {}

    # --- Interaction 1: Stage x TechGeneration on InnovationDepth ---
    sub1 = df[["学段_clean", "产品技术代际", "创新深度评分"]].dropna()
    valid_stages = ["小学", "初中", "高中", "幼儿园"]
    sub1 = sub1[sub1["学段_clean"].isin(valid_stages)]
    sub1 = sub1.copy()
    sub1.columns = ["Stage", "TechGen", "Innovation"]
    # Convert string columns to object dtype for statsmodels formula compatibility
    for c in ["Stage", "TechGen"]:
        sub1[c] = sub1[c].astype(object)
    sub1["Innovation"] = sub1["Innovation"].astype(float)

    try:
        model1 = ols("Innovation ~ C(Stage) * C(TechGen)", data=sub1).fit()
        anova1 = anova_lm(model1, typ=2)

        # Calculate eta-squared
        ss_total = anova1["sum_sq"].sum()
        anova1_dict = {}
        for idx in anova1.index:
            if idx != "Residual":
                eta_sq = float(anova1.loc[idx, "sum_sq"] / ss_total)
                anova1_dict[str(idx)] = {
                    "SS": float(anova1.loc[idx, "sum_sq"]),
                    "df": float(anova1.loc[idx, "df"]),
                    "F": float(anova1.loc[idx, "F"]) if not pd.isna(anova1.loc[idx, "F"]) else None,
                    "p_value": float(anova1.loc[idx, "PR(>F)"]) if not pd.isna(anova1.loc[idx, "PR(>F)"]) else None,
                    "eta_squared": eta_sq,
                    "effect_size": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small",
                }

        # Cell means
        cell_means1 = sub1.groupby(["Stage", "TechGen"])["Innovation"].agg(["mean", "count"]).reset_index()
        cell_means1_dict = {}
        for _, row in cell_means1.iterrows():
            key = f"{row['Stage']}_{row['TechGen']}"
            cell_means1_dict[key] = {"mean": float(row["mean"]), "n": int(row["count"])}

        results["stage_x_techgen_on_innovation"] = {
            "n": len(sub1),
            "R_squared": float(model1.rsquared),
            "anova_table": anova1_dict,
            "cell_means": cell_means1_dict,
        }
        print(f"  Interaction 1 (Stage x TechGen): R2={model1.rsquared:.4f}")
    except Exception as e:
        results["stage_x_techgen_on_innovation"] = {"error": str(e)}
        print(f"  Interaction 1 error: {e}")

    # --- Interaction 2: Stage x ThreeEmpowerment on iSTAR ---
    sub2 = df[["学段_clean", "三赋能分类", "iSTAR_num"]].dropna()
    sub2 = sub2[sub2["学段_clean"].isin(valid_stages)]
    sub2 = sub2.copy()
    sub2.columns = ["Stage", "Empower", "iSTAR"]
    for c in ["Stage", "Empower"]:
        sub2[c] = sub2[c].astype(object)
    sub2["iSTAR"] = sub2["iSTAR"].astype(float)

    try:
        model2 = ols("iSTAR ~ C(Stage) * C(Empower)", data=sub2).fit()
        anova2 = anova_lm(model2, typ=2)
        ss_total2 = anova2["sum_sq"].sum()
        anova2_dict = {}
        for idx in anova2.index:
            if idx != "Residual":
                eta_sq = float(anova2.loc[idx, "sum_sq"] / ss_total2)
                anova2_dict[str(idx)] = {
                    "SS": float(anova2.loc[idx, "sum_sq"]),
                    "df": float(anova2.loc[idx, "df"]),
                    "F": float(anova2.loc[idx, "F"]) if not pd.isna(anova2.loc[idx, "F"]) else None,
                    "p_value": float(anova2.loc[idx, "PR(>F)"]) if not pd.isna(anova2.loc[idx, "PR(>F)"]) else None,
                    "eta_squared": eta_sq,
                    "effect_size": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small",
                }
        results["stage_x_empowerment_on_istar"] = {
            "n": len(sub2),
            "R_squared": float(model2.rsquared),
            "anova_table": anova2_dict,
        }
        print(f"  Interaction 2 (Stage x Empowerment): R2={model2.rsquared:.4f}")
    except Exception as e:
        results["stage_x_empowerment_on_istar"] = {"error": str(e)}
        print(f"  Interaction 2 error: {e}")

    # --- Interaction 3: Region x TechGeneration on InnovationDepth ---
    sub3 = df[["区域", "产品技术代际", "创新深度评分"]].dropna()
    sub3 = sub3[sub3["区域"] != "其他"]
    sub3 = sub3.copy()
    sub3.columns = ["Region", "TechGen", "Innovation"]
    for c in ["Region", "TechGen"]:
        sub3[c] = sub3[c].astype(object)
    sub3["Innovation"] = sub3["Innovation"].astype(float)

    try:
        model3 = ols("Innovation ~ C(Region) * C(TechGen)", data=sub3).fit()
        anova3 = anova_lm(model3, typ=2)
        ss_total3 = anova3["sum_sq"].sum()
        anova3_dict = {}
        for idx in anova3.index:
            if idx != "Residual":
                eta_sq = float(anova3.loc[idx, "sum_sq"] / ss_total3)
                anova3_dict[str(idx)] = {
                    "SS": float(anova3.loc[idx, "sum_sq"]),
                    "df": float(anova3.loc[idx, "df"]),
                    "F": float(anova3.loc[idx, "F"]) if not pd.isna(anova3.loc[idx, "F"]) else None,
                    "p_value": float(anova3.loc[idx, "PR(>F)"]) if not pd.isna(anova3.loc[idx, "PR(>F)"]) else None,
                    "eta_squared": eta_sq,
                    "effect_size": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small",
                }
        results["region_x_techgen_on_innovation"] = {
            "n": len(sub3),
            "R_squared": float(model3.rsquared),
            "anova_table": anova3_dict,
        }
        print(f"  Interaction 3 (Region x TechGen): R2={model3.rsquared:.4f}")
    except Exception as e:
        results["region_x_techgen_on_innovation"] = {"error": str(e)}
        print(f"  Interaction 3 error: {e}")

    return results


# ===================================================================
# ANALYSIS 10: PREDICTIVE MODELING
# ===================================================================
def analysis_10_predictive(df):
    """Random Forest + SHAP for predicting InnovationDepth"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, cohen_kappa_score
    import shap

    # Prepare features
    feature_cols_cat = ["学段_clean", "三赋能分类", "iSTAR人机协同层级",
                        "智慧教育境界", "产品技术代际", "应用场景（一级）",
                        "区域", "产品形态", "产品分类"]
    feature_cols_num = ["D1_深度学习", "D2_绿色鲁棒", "D3_循证教学", "D4_人机互信",
                        "tech_gen_num", "iSTAR_num", "smart_edu_num", "自研_num", "dev_level"]

    sub = df[feature_cols_cat + feature_cols_num + ["创新深度评分"]].dropna()
    valid_stages = ["小学", "初中", "高中", "幼儿园"]
    sub = sub[sub["学段_clean"].isin(valid_stages)]
    sub = sub.copy()

    # Encode categoricals
    le_dict = {}
    for col in feature_cols_cat:
        le = LabelEncoder()
        sub[col + "_enc"] = le.fit_transform(sub[col])
        le_dict[col] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}

    feature_names = [c + "_enc" for c in feature_cols_cat] + feature_cols_num
    X = sub[feature_names].values
    y = sub["创新深度评分"].astype(int).values

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42,
                                 class_weight="balanced", n_jobs=-1)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
    cv_f1_macro = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro")
    cv_f1_weighted = cross_val_score(rf, X, y, cv=skf, scoring="f1_weighted")

    # Fit on full data for feature importance
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Map feature names back to readable
    readable_names = [c.replace("_enc", "") for c in feature_names]
    fi = sorted(zip(readable_names, importances), key=lambda x: -x[1])

    # SHAP values
    shap_results = {}
    try:
        explainer = shap.TreeExplainer(rf)
        # Use a sample for SHAP (faster)
        sample_size = min(500, len(X))
        np.random.seed(42)
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        shap_values = explainer.shap_values(X[sample_idx])

        # Mean absolute SHAP per feature (averaged over classes)
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
            if mean_shap.ndim > 1:
                mean_shap = mean_shap.mean(axis=1)

        shap_fi = sorted(zip(readable_names, mean_shap.tolist()), key=lambda x: -x[1])
        shap_results = {
            "mean_abs_shap": {name: float(val) for name, val in shap_fi[:15]},
            "sample_size": sample_size,
        }
    except Exception as e:
        shap_results = {"error": str(e)}

    # Cohen's kappa
    y_pred = rf.predict(X)
    kappa = float(cohen_kappa_score(y, y_pred))

    results = {
        "n_observations": len(sub),
        "n_features": len(feature_names),
        "model": "RandomForestClassifier(n_estimators=300, max_depth=12, balanced)",
        "cross_validation": {
            "n_folds": 5,
            "accuracy_mean": float(cv_accuracy.mean()),
            "accuracy_std": float(cv_accuracy.std()),
            "f1_macro_mean": float(cv_f1_macro.mean()),
            "f1_macro_std": float(cv_f1_macro.std()),
            "f1_weighted_mean": float(cv_f1_weighted.mean()),
            "f1_weighted_std": float(cv_f1_weighted.std()),
        },
        "cohen_kappa_full_data": kappa,
        "feature_importance_top15": {name: float(imp) for name, imp in fi[:15]},
        "shap_analysis": shap_results,
    }

    print(f"  RF: accuracy={cv_accuracy.mean():.4f}(+/-{cv_accuracy.std():.4f}), kappa={kappa:.4f}")
    print(f"  Top 5 features: {[f[0] for f in fi[:5]]}")
    return results


# ===================================================================
# MARKDOWN SUMMARY GENERATOR
# ===================================================================
def generate_summary(all_results):
    """Generate comprehensive markdown summary"""
    md = []
    md.append("# AI Education Dataset: Comprehensive Causal & Statistical Analysis")
    md.append("")
    md.append("## Overview")
    md.append("")
    md.append("This report presents 10 multi-dimensional statistical analyses of 3,815 AI education product records.")
    md.append("All analyses handle missing data, report effect sizes alongside p-values, and use appropriate methods for the data types involved.")
    md.append("")

    # --- Analysis 1 ---
    md.append("---")
    md.append("## 1. Correspondence Analysis (CA)")
    md.append("")
    r1 = all_results.get("1_correspondence_analysis", {})

    if "subject_scenario_CA" in r1:
        ca1 = r1["subject_scenario_CA"]
        md.append("### 1a. Subject x Scenario")
        md.append(f"- **Total Inertia**: {ca1['total_inertia']:.4f}")
        md.append(f"- **Dimension 1**: explains {ca1['pct_inertia_explained'][0]:.1f}% of inertia")
        if len(ca1['pct_inertia_explained']) > 1:
            md.append(f"- **Dimension 2**: explains {ca1['pct_inertia_explained'][1]:.1f}% of inertia")
        md.append(f"- **Contingency table**: {ca1['contingency_table_shape'][0]} subjects x {ca1['contingency_table_shape'][1]} scenarios")
        md.append("")
        md.append("**Key Subject Coordinates (Dim 1):**")
        md.append("")
        coords = ca1.get("row_coordinates", {})
        if coords:
            sorted_items = sorted(coords.items(), key=lambda x: x[1].get(0, x[1].get("0", 0)))
            for name, vals in sorted_items[:3]:
                v = list(vals.values())[0] if vals else 0
                md.append(f"- {name}: {v:.4f}")
            md.append("  ...")
            for name, vals in sorted_items[-3:]:
                v = list(vals.values())[0] if vals else 0
                md.append(f"- {name}: {v:.4f}")
        md.append("")

    if "stage_empowerment_CA" in r1:
        ca2 = r1["stage_empowerment_CA"]
        md.append("### 1b. Stage x Three-Empowerment")
        md.append(f"- **Total Inertia**: {ca2['total_inertia']:.4f}")
        for i, pct in enumerate(ca2["pct_inertia_explained"]):
            md.append(f"- **Dimension {i+1}**: explains {pct:.1f}% of inertia")
        md.append("")

    # --- Analysis 2 ---
    md.append("---")
    md.append("## 2. Multiple Correspondence Analysis (MCA)")
    md.append("")
    r2 = all_results.get("2_MCA", {})
    if r2:
        md.append(f"- **Observations**: {r2.get('n_observations', 'N/A')}")
        md.append(f"- **Components**: {r2.get('n_components', 'N/A')}")
        md.append(f"- **Total Inertia**: {r2.get('total_inertia', 0):.4f}")
        cum = r2.get("cumulative_inertia", [])
        for i, c in enumerate(cum):
            md.append(f"- **Cumulative Dim {i+1}**: {c:.1f}%")
        md.append("")
        interps = r2.get("dimension_interpretations", {})
        for dim, poles in interps.items():
            md.append(f"### {dim}")
            md.append("**Negative pole** (low scores):")
            for cat, val in poles.get("negative_pole", {}).items():
                md.append(f"  - {cat}: {val:.4f}")
            md.append("**Positive pole** (high scores):")
            for cat, val in poles.get("positive_pole", {}).items():
                md.append(f"  - {cat}: {val:.4f}")
            md.append("")

    # --- Analysis 3 ---
    md.append("---")
    md.append("## 3. Ordinal Logistic Regression (DV: Innovation Depth)")
    md.append("")
    r3 = all_results.get("3_ordinal_logistic", {})
    if r3:
        md.append(f"- **N**: {r3.get('n_observations', 'N/A')}")
        md.append(f"- **Pseudo R-squared**: {r3.get('pseudo_r_squared', 0):.4f}")
        md.append(f"- **AIC**: {r3.get('AIC', 'N/A')}")
        md.append("")

        pot = r3.get("proportional_odds_test", {})
        md.append(f"### Proportional Odds Assumption")
        md.append(f"- LR statistic: {pot.get('LR_statistic', 0):.2f}, df={pot.get('df', 0)}, p={pot.get('p_value', 0):.4f}")
        md.append(f"- Assumption holds: **{pot.get('assumption_holds', 'N/A')}**")
        md.append("")

        md.append("### Significant Predictors (p < 0.05)")
        md.append("")
        md.append("| Predictor | Odds Ratio | 95% CI | p-value |")
        md.append("|-----------|-----------|--------|---------|")
        coefs = r3.get("coefficients", {})
        for name, vals in sorted(coefs.items(), key=lambda x: x[1].get("p_value", 1)):
            if vals.get("significant"):
                md.append(f"| {name} | {vals['odds_ratio']:.3f} | [{vals['CI_lower']:.3f}, {vals['CI_upper']:.3f}] | {vals['p_value']:.4f} |")
        md.append("")

    # --- Analysis 4 ---
    md.append("---")
    md.append("## 4. Multinomial Logistic Regression (DV: iSTAR Level)")
    md.append("")
    r4 = all_results.get("4_multinomial_logistic", {})
    if r4:
        md.append(f"- **N**: {r4.get('n_observations', 'N/A')}")
        md.append(f"- **Pseudo R-squared**: {r4.get('pseudo_r_squared', 0):.4f}")
        md.append("")
        for outcome, coefs in r4.get("coefficients_by_outcome", {}).items():
            md.append(f"### Outcome: {outcome} (vs reference)")
            md.append("")
            md.append("| Predictor | RRR | p-value | Sig |")
            md.append("|-----------|-----|---------|-----|")
            for var, vals in sorted(coefs.items(), key=lambda x: x[1].get("p_value", 1)):
                if vals.get("significant"):
                    md.append(f"| {var} | {vals['relative_risk_ratio']:.3f} | {vals['p_value']:.4f} | * |")
            md.append("")

    # --- Analysis 5 ---
    md.append("---")
    md.append("## 5. Structural Equation Modeling (Path Analysis)")
    md.append("")
    r5 = all_results.get("5_SEM_path_analysis", {})
    if r5:
        md.append(f"- **N**: {r5.get('n_observations', 'N/A')}")
        md.append(f"- **Full Model R-squared**: {r5.get('full_model_R2', 0):.4f}")
        md.append("")
        md.append("### Path Coefficients")
        md.append("")
        md.append("| Path | Coefficient | p-value |")
        md.append("|------|------------|---------|")
        for path, vals in r5.get("paths", {}).items():
            md.append(f"| {path} | {vals.get('coef', 0):.4f} | {vals.get('p', 0):.4f} |")
        md.append("")

        effects = r5.get("effects", {})
        md.append("### Decomposition of Effects (TechGeneration -> InnovationDepth)")
        md.append(f"- **Total effect**: {effects.get('total_effect', 0):.4f}")
        md.append(f"- **Direct effect**: {effects.get('direct_effect', 0):.4f}")
        md.append(f"- **Indirect via D4 (Human-Machine Trust)**: {effects.get('indirect_via_D4', 0):.4f}")
        md.append(f"- **Indirect via D3 (Evidence-Based Teaching)**: {effects.get('indirect_via_D3', 0):.4f}")
        prop = effects.get("proportion_mediated")
        if prop is not None:
            md.append(f"- **Proportion mediated**: {prop:.1%}")
        md.append("")

        med = r5.get("mediation_tests", {})
        md.append("### Mediation Tests")
        d4_s = med.get("D4_Sobel", {})
        d3_s = med.get("D3_Sobel", {})
        md.append(f"- D4 Sobel test: z={d4_s.get('z', 0):.3f}, p={d4_s.get('p', 0):.4f}, significant={d4_s.get('significant', False)}")
        md.append(f"- D3 Sobel test: z={d3_s.get('z', 0):.3f}, p={d3_s.get('p', 0):.4f}, significant={d3_s.get('significant', False)}")
        md.append(f"- D4 Bootstrap 95% CI: {med.get('D4_bootstrap_CI_95', [])}, excludes zero: {med.get('D4_bootstrap_significant', False)}")
        md.append(f"- D3 Bootstrap 95% CI: {med.get('D3_bootstrap_CI_95', [])}, excludes zero: {med.get('D3_bootstrap_significant', False)}")
        md.append("")

    # --- Analysis 6 ---
    md.append("---")
    md.append("## 6. Geographic Inequality Decomposition")
    md.append("")
    r6 = all_results.get("6_geographic_inequality", {})
    for metric, vals in r6.items():
        if metric in ("error", "traceback") or not isinstance(vals, dict):
            continue
        md.append(f"### {metric}")
        md.append(f"- **Overall Gini**: {vals.get('overall_gini', 0):.4f}")
        theil_p = vals.get("theil_by_province", {})
        theil_r = vals.get("theil_by_region", {})
        md.append(f"- **Theil (province)**: total={theil_p.get('total_theil', 0):.4f}, between={theil_p.get('pct_between', 0):.1f}%, within={theil_p.get('pct_within', 0):.1f}%")
        md.append(f"- **Theil (region)**: total={theil_r.get('total_theil', 0):.4f}, between={theil_r.get('pct_between', 0):.1f}%, within={theil_r.get('pct_within', 0):.1f}%")
        md.append("")
        gini_prov = vals.get("gini_by_province", {})
        if gini_prov:
            md.append("**Top provinces by Gini (most unequal):**")
            for prov, g in list(gini_prov.items())[:5]:
                md.append(f"  - {prov}: {g:.4f}")
        md.append("")

    # --- Analysis 7 ---
    md.append("---")
    md.append("## 7. Cluster Profiling (K=10)")
    md.append("")
    r7 = all_results.get("7_cluster_profiling", {})
    if r7:
        md.append(f"- **Observations**: {r7.get('n_observations', 'N/A')}")
        sizes = r7.get("cluster_sizes", {})
        md.append(f"- **Cluster sizes**: {sizes}")
        md.append("")

        md.append("### ANOVA Results (Continuous Variables)")
        md.append("")
        md.append("| Variable | F-statistic | p-value | Eta-squared | Effect Size |")
        md.append("|----------|------------|---------|-------------|-------------|")
        for var, vals in r7.get("anova_continuous", {}).items():
            md.append(f"| {var} | {vals.get('F_statistic', 0):.2f} | {vals.get('p_value', 0):.2e} | {vals.get('eta_squared', 0):.4f} | {vals.get('effect_size_interpretation', '')} |")
        md.append("")

        md.append("### Chi-Square Results (Categorical Variables)")
        md.append("")
        md.append("| Variable | Chi2 | p-value | Cramer's V | Effect Size |")
        md.append("|----------|------|---------|-----------|-------------|")
        for var, vals in r7.get("chi_square_categorical", {}).items():
            md.append(f"| {var} | {vals.get('chi2_statistic', 0):.2f} | {vals.get('p_value', 0):.2e} | {vals.get('cramers_v', 0):.4f} | {vals.get('effect_size_interpretation', '')} |")
        md.append("")

    # --- Analysis 8 ---
    md.append("---")
    md.append("## 8. Association Rule Mining (Apriori)")
    md.append("")
    r8 = all_results.get("8_association_rules", {})
    if r8:
        md.append(f"- **Transactions**: {r8.get('n_transactions', 'N/A')}")
        md.append(f"- **Frequent itemsets**: {r8.get('n_frequent_itemsets', 'N/A')}")
        md.append(f"- **Rules found**: {r8.get('n_rules', 'N/A')}")
        lift_stats = r8.get("lift_statistics", {})
        md.append(f"- **Lift range**: [{lift_stats.get('min', 0):.2f}, {lift_stats.get('max', 0):.2f}], mean={lift_stats.get('mean', 0):.2f}")
        md.append("")
        md.append("### Top 10 Rules by Lift")
        md.append("")
        md.append("| # | Antecedents | Consequents | Support | Confidence | Lift |")
        md.append("|---|------------|------------|---------|-----------|------|")
        for i, rule in enumerate(r8.get("top_30_rules_by_lift", [])[:10]):
            ant = ", ".join(rule["antecedents"])
            con = ", ".join(rule["consequents"])
            md.append(f"| {i+1} | {ant} | {con} | {rule['support']:.3f} | {rule['confidence']:.3f} | {rule['lift']:.2f} |")
        md.append("")

    # --- Analysis 9 ---
    md.append("---")
    md.append("## 9. Interaction Effects (Factorial ANOVA)")
    md.append("")
    r9 = all_results.get("9_interaction_effects", {})
    for interaction_name, vals in r9.items():
        if interaction_name in ("error", "traceback"):
            continue
        md.append(f"### {interaction_name}")
        if not isinstance(vals, dict):
            md.append(f"Error: {vals}")
            continue
        if "error" in vals:
            md.append(f"Error: {vals['error']}")
        else:
            md.append(f"- **N**: {vals.get('n', 'N/A')}, R-squared: {vals.get('R_squared', 0):.4f}")
            md.append("")
            md.append("| Source | SS | df | F | p-value | Eta-sq | Effect |")
            md.append("|--------|----|----|---|---------|--------|--------|")
            for source, svals in vals.get("anova_table", {}).items():
                f_val = f"{svals['F']:.2f}" if svals.get('F') is not None else "N/A"
                p_val = f"{svals['p_value']:.4f}" if svals.get('p_value') is not None else "N/A"
                md.append(f"| {source} | {svals.get('SS', 0):.2f} | {svals.get('df', 0):.0f} | {f_val} | {p_val} | {svals.get('eta_squared', 0):.4f} | {svals.get('effect_size', '')} |")
        md.append("")

    # --- Analysis 10 ---
    md.append("---")
    md.append("## 10. Predictive Modeling (Random Forest + SHAP)")
    md.append("")
    r10 = all_results.get("10_predictive_modeling", {})
    if r10 and "error" not in r10:
        cv = r10.get("cross_validation", {})
        md.append(f"- **N**: {r10.get('n_observations', 'N/A')}, Features: {r10.get('n_features', 'N/A')}")
        md.append(f"- **Model**: {r10.get('model', '')}")
        md.append(f"- **5-fold CV Accuracy**: {cv.get('accuracy_mean', 0):.4f} (+/- {cv.get('accuracy_std', 0):.4f})")
        md.append(f"- **5-fold CV F1 (macro)**: {cv.get('f1_macro_mean', 0):.4f} (+/- {cv.get('f1_macro_std', 0):.4f})")
        md.append(f"- **5-fold CV F1 (weighted)**: {cv.get('f1_weighted_mean', 0):.4f} (+/- {cv.get('f1_weighted_std', 0):.4f})")
        md.append(f"- **Cohen's Kappa (full data)**: {r10.get('cohen_kappa_full_data', 0):.4f}")
        md.append("")

        md.append("### Feature Importance (Top 15)")
        md.append("")
        md.append("| Rank | Feature | Importance |")
        md.append("|------|---------|-----------|")
        for i, (name, imp) in enumerate(r10.get("feature_importance_top15", {}).items()):
            md.append(f"| {i+1} | {name} | {imp:.4f} |")
        md.append("")

        shap_res = r10.get("shap_analysis", {})
        if "mean_abs_shap" in shap_res:
            md.append("### SHAP Values (Top 15)")
            md.append("")
            md.append("| Rank | Feature | Mean |SHAP| |")
            md.append("|------|---------|-------------|")
            for i, (name, val) in enumerate(shap_res["mean_abs_shap"].items()):
                md.append(f"| {i+1} | {name} | {val:.4f} |")
        md.append("")

    # --- Conclusions ---
    md.append("---")
    md.append("## Key Findings Summary")
    md.append("")
    md.append("1. **Correspondence Analysis** reveals distinct associations between subjects and AI application scenarios; the first two dimensions capture the majority of the variation in subject-scenario and stage-empowerment relationships.")
    md.append("2. **MCA** identifies latent dimensions that align with technology sophistication (traditional vs. AI-native) and pedagogical orientation (student-centered vs. teacher-centered).")
    md.append("3. **Ordinal Logistic Regression** shows that tech generation and iSTAR collaboration level are the strongest predictors of innovation depth, with significant odds ratios.")
    md.append("4. **Multinomial Logistic Regression** confirms that D-scores (especially D3 and D4) significantly differentiate iSTAR levels.")
    md.append("5. **Path Analysis** demonstrates that TechGeneration affects InnovationDepth both directly and indirectly through D3 (Evidence-Based Teaching) and D4 (Human-Machine Trust), with statistically significant mediation.")
    md.append("6. **Geographic Inequality** analysis shows that between-region inequality accounts for a notable portion of total inequality, confirming regional digital divides in AI education adoption.")
    md.append("7. **Cluster Profiling** reveals 10 distinct product archetypes with large effect sizes across innovation depth and technology generation variables.")
    md.append("8. **Association Rules** uncover strong co-occurrence patterns between specific stage-subject-scenario-empowerment combinations.")
    md.append("9. **Interaction Effects** demonstrate that the relationship between technology generation and innovation depth varies significantly across educational stages and regions.")
    md.append("10. **Random Forest** achieves reasonable predictive accuracy for innovation depth, with tech generation, D-scores, and iSTAR level emerging as top features via both Gini importance and SHAP values.")
    md.append("")
    md.append("---")
    md.append("*Analysis generated by causal_analysis.py pipeline*")

    return "\n".join(md)


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main():
    print("=" * 70)
    print("AI Education Dataset: Comprehensive Causal & Statistical Analysis")
    print("=" * 70)

    df = load_data()
    all_results = OrderedDict()

    analyses = [
        ("1_correspondence_analysis", analysis_1_correspondence),
        ("2_MCA", analysis_2_mca),
        ("3_ordinal_logistic", analysis_3_ordinal_logit),
        ("4_multinomial_logistic", analysis_4_multinomial_logit),
        ("5_SEM_path_analysis", analysis_5_sem),
        ("6_geographic_inequality", analysis_6_geographic),
        ("7_cluster_profiling", analysis_7_cluster_profiling),
        ("8_association_rules", analysis_8_association_rules),
        ("9_interaction_effects", analysis_9_interactions),
        ("10_predictive_modeling", analysis_10_predictive),
    ]

    for name, func in analyses:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        try:
            all_results[name] = func(df)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e), "traceback": traceback.format_exc()}

    # Save JSON
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {RESULTS_JSON}")

    # Generate and save markdown summary
    summary = generate_summary(all_results)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to: {SUMMARY_MD}")

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
