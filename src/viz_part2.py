#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_part2.py  --  Advanced Statistical & Causal Figures (Part 2)
Generates 12 publication-quality figures for the AI-in-Education research project.
"""

import warnings
warnings.filterwarnings("ignore")

import json, ast, re, textwrap
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform

import networkx as nx
from networkx.algorithms.community import louvain_communities

# ---------- paths -----------------------------------------------------------
BASE      = Path("/Users/sakai/Desktop/产业调研/ai-edu-research")
CSV_PATH  = BASE / "output" / "教育产品统计_V6_框架标注.csv"
JSON_PATH = BASE / "output" / "causal_analysis_results.json"
FIG_DIR   = BASE / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load data -------------------------------------------------------
df = pd.read_csv(CSV_PATH)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    causal = json.load(f)

# ---------- global style ----------------------------------------------------
CN_FONT = "Source Han Sans CN"
EN_FONT = "Helvetica Neue"
plt.rcParams.update({
    "font.family":       [CN_FONT, EN_FONT, "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi":         150,
    "savefig.dpi":        400,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.3,
})

DPI = 400

# -- Rich color palettes --
PAL_MAIN   = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261",
              "#264653","#A8DADC","#6D6875","#B5838D","#FFCDB2"]
PAL_WARM   = ["#D62828","#F77F00","#FCBF49","#EAE2B7","#003049"]
PAL_COOL   = ["#03045E","#0077B6","#00B4D8","#90E0EF","#CAF0F8"]
PAL_VIVID  = ["#FF006E","#8338EC","#3A86FF","#FB5607","#FFBE0B",
              "#06D6A0","#118AB2","#EF476F","#073B4C","#FFD166"]
PAL_EARTH  = ["#606C38","#283618","#FEFAE0","#DDA15E","#BC6C25"]
PAL_BOLD   = ["#7400B8","#6930C3","#5390D9","#48BFE3","#56CFE1",
              "#64DFDF","#72EFDD","#80FFDB"]


def add_source_note(fig, fig_num, extra=""):
    """Add figure number and source footnote at the bottom."""
    txt = f"Figure {fig_num} | Data: AI-Education Product Survey 2026 (N={len(df):,})"
    if extra:
        txt += f"  |  {extra}"
    fig.text(0.5, 0.005, txt,
             ha="center", va="bottom", fontsize=8,
             color="#555555", fontstyle="italic",
             fontfamily=EN_FONT)


def add_annotation_box(ax, text, xy, xytext, fontsize=9,
                       boxcolor="#FFF3CD", edgecolor="#FFC107", alpha=0.95):
    """Add a callout annotation box with arrow."""
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=fontsize, fontfamily=CN_FONT,
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc=boxcolor, ec=edgecolor,
                          alpha=alpha, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=edgecolor,
                                connectionstyle="arc3,rad=0.2", lw=1.5))


def add_text_box(ax, text, x, y, fontsize=9,
                 boxcolor="#E8F4FD", edgecolor="#2196F3", alpha=0.92,
                 transform=None):
    """Add a standalone text box (no arrow)."""
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, transform=transform,
            fontsize=fontsize, fontfamily=CN_FONT,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.6", fc=boxcolor, ec=edgecolor,
                      alpha=alpha, lw=1.2))


# ============================================================================
# F02  Cramer's V Correlation Matrix with Dendrogram
# ============================================================================
def fig_f02():
    print("  [1/12] fig_f02_cramers_v ...")
    cat_cols = ["学段","应用场景（一级）","三赋能分类","iSTAR人机协同层级",
                "智慧教育境界","产品技术代际","产品分类","产品形态","学科"]
    sub = df[cat_cols].dropna()

    n = len(cat_cols)
    V = np.zeros((n, n))
    P = np.ones((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                V[i, j] = 1.0
                P[i, j] = 0.0
            else:
                ct = pd.crosstab(sub[cat_cols[i]], sub[cat_cols[j]])
                chi2, p, _, _ = stats.chi2_contingency(ct)
                k = min(ct.shape) - 1
                nobs = ct.values.sum()
                v = np.sqrt(chi2 / (nobs * k)) if k > 0 else 0
                V[i, j] = V[j, i] = v
                P[i, j] = P[j, i] = p

    # hierarchical clustering to reorder
    dist = 1 - V
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[1, 4],
                           hspace=0.02, wspace=0.02)

    # dendrogram on top
    ax_dendro_top = fig.add_subplot(gs[0, 1])
    dn = dendrogram(Z, no_labels=True, color_threshold=0.5,
                    above_threshold_color="#999999", ax=ax_dendro_top)
    ax_dendro_top.set_xticks([])
    ax_dendro_top.set_yticks([])
    for spine in ax_dendro_top.spines.values():
        spine.set_visible(False)

    order = dn["leaves"]
    V_ordered = V[np.ix_(order, order)]
    P_ordered = P[np.ix_(order, order)]
    labels_ordered = [cat_cols[i] for i in order]

    # dendrogram on left
    ax_dendro_left = fig.add_subplot(gs[1, 0])
    dendrogram(Z, no_labels=True, orientation="left",
               color_threshold=0.5,
               above_threshold_color="#999999", ax=ax_dendro_left)
    ax_dendro_left.set_xticks([])
    ax_dendro_left.set_yticks([])
    for spine in ax_dendro_left.spines.values():
        spine.set_visible(False)
    ax_dendro_left.invert_yaxis()

    # heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    cmap = plt.cm.RdYlGn_r
    im = ax_heat.imshow(V_ordered, cmap=cmap, aspect="auto",
                        vmin=0, vmax=0.6)

    # significance stars
    for i in range(n):
        for j in range(n):
            if i != j:
                val = V_ordered[i, j]
                p = P_ordered[i, j]
                star = ""
                if p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
                txt_color = "white" if val > 0.35 else "black"
                ax_heat.text(j, i, f"{val:.2f}\n{star}", ha="center", va="center",
                             fontsize=8, fontweight="bold", color=txt_color)

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=10)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(labels_ordered, fontsize=10)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.02)
    cbar.set_label("Cramer's V", fontsize=11, fontfamily=EN_FONT)

    fig.suptitle("Cramer's V 相关矩阵（层次聚类排序）\nCramer's V Correlation Matrix with Hierarchical Clustering",
                 fontsize=16, fontweight="bold", y=0.98)

    # annotation box
    add_text_box(ax_heat,
                 "Cramer's V 衡量分类变量间的关联强度\n"
                 "0.1=弱  0.3=中等  0.5=强\n"
                 "*** p<0.001  ** p<0.01  * p<0.05\n"
                 "聚类树状图揭示变量间结构相似性",
                 0.02, -0.18, fontsize=9,
                 boxcolor="#FFF8E1", edgecolor="#FF8F00")

    ax_empty = fig.add_subplot(gs[0, 0])
    ax_empty.axis("off")

    add_source_note(fig, "F02", "Cramer's V with Ward clustering")
    fig.savefig(FIG_DIR / "fig_f02_cramers_v.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# F03  Correspondence Analysis Biplot
# ============================================================================
def fig_f03():
    print("  [2/12] fig_f03_correspondence_biplot ...")
    ca = causal["1_correspondence_analysis"]["subject_scenario_CA"]
    row_coords = ca["row_coordinates"]
    col_coords = ca["col_coordinates"]
    pct = ca["pct_inertia_explained"]

    fig, ax = plt.subplots(figsize=(16, 11))

    # scenario colors
    scen_colors = {"助学":"#E63946","助教":"#457B9D","助评":"#2A9D8F",
                   "助管":"#E9C46A","助育":"#F4A261","助研":"#264653"}

    # Draw subjects
    subj_x = [v["0"] for v in row_coords.values()]
    subj_y = [v["1"] for v in row_coords.values()]
    subj_names = list(row_coords.keys())

    ax.scatter(subj_x, subj_y, s=60, c="#6D6875", alpha=0.7,
              zorder=3, edgecolors="white", linewidths=0.5)
    for name, x, y in zip(subj_names, subj_x, subj_y):
        if abs(x) > 0.3 or abs(y) > 0.4:
            ax.annotate(name, (x, y), fontsize=8, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points",
                        fontweight="bold", color="#333333")
        else:
            ax.annotate(name, (x, y), fontsize=7, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points",
                        color="#666666")

    # Draw scenarios as vectors
    for scen, coord in col_coords.items():
        x, y = coord["0"], coord["1"]
        color = scen_colors.get(scen, "#999999")
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.5, mutation_scale=15))
        ax.text(x * 1.08, y * 1.08, scen, fontsize=12, fontweight="bold",
                color=color, ha="center", va="center",
                bbox=dict(fc="white", ec=color, alpha=0.85, boxstyle="round,pad=0.3"))

    # Convex hulls for major scenarios
    # Group subjects by closest scenario
    for scen, coord in col_coords.items():
        sx, sy = coord["0"], coord["1"]
        # find subjects within reasonable proximity
        nearby = []
        for name in subj_names:
            rx, ry = row_coords[name]["0"], row_coords[name]["1"]
            d = np.sqrt((rx - sx)**2 + (ry - sy)**2)
            if d < 1.5:
                nearby.append((rx, ry))
        if len(nearby) >= 3:
            pts = np.array(nearby)
            try:
                hull = ConvexHull(pts)
                color = scen_colors.get(scen, "#999999")
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        alpha=0.06, color=color)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        '--', alpha=0.3, color=color, lw=1)
            except:
                pass

    ax.axhline(0, color="#CCCCCC", lw=0.8, ls="--")
    ax.axvline(0, color="#CCCCCC", lw=0.8, ls="--")

    ax.set_xlabel(f"Dimension 1 ({pct[0]:.1f}% inertia)", fontsize=13, fontfamily=EN_FONT)
    ax.set_ylabel(f"Dimension 2 ({pct[1]:.1f}% inertia)", fontsize=13, fontfamily=EN_FONT)
    ax.set_title("学科 x 应用场景 对应分析双标图\nCorrespondence Analysis Biplot: Subject x Scenario",
                 fontsize=16, fontweight="bold", pad=15)

    add_text_box(ax,
                 "箭头方向=场景定位, 箭头长度=辨别力\n"
                 "助学场景集中于核心学科(数语英)\n"
                 "助育/助评距核心远, 具有独特学科组合\n"
                 f"总惯量解释: {pct[0]+pct[1]:.1f}%",
                 0.01, 0.99, fontsize=9,
                 boxcolor="#E8F5E9", edgecolor="#4CAF50")

    add_source_note(fig, "F03", "CA biplot, subject x scenario")
    fig.savefig(FIG_DIR / "fig_f03_correspondence_biplot.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# F04  MCA Biplot
# ============================================================================
def fig_f04():
    print("  [3/12] fig_f04_mca_biplot ...")
    mca = causal["2_MCA"]
    col_coords = mca["column_coordinates_factor_loadings"]
    pct = mca["pct_inertia_explained"]

    # parse category types
    cat_types = {}
    cat_type_colors = {
        "学段_clean": "#E63946",
        "三赋能分类": "#457B9D",
        "iSTAR人机协同层级": "#2A9D8F",
        "智慧教育境界": "#E9C46A",
        "产品技术代际": "#F4A261",
    }

    fig, ax = plt.subplots(figsize=(16, 11))

    for fullname, coords in col_coords.items():
        parts = fullname.split("__")
        cat_type = parts[0]
        label = parts[1] if len(parts) > 1 else fullname
        x, y = coords["0"], coords["1"]
        color = cat_type_colors.get(cat_type, "#6D6875")

        # Skip extreme outliers for visual clarity
        if abs(x) > 5 or abs(y) > 5:
            # still plot but smaller
            ax.scatter(x, y, s=40, c=color, alpha=0.5, zorder=3,
                       marker="D", edgecolors="white")
            ax.annotate(label, (x, y), fontsize=7, color=color, alpha=0.6,
                        xytext=(4, 4), textcoords="offset points")
        else:
            ax.scatter(x, y, s=120, c=color, alpha=0.8, zorder=3,
                       edgecolors="white", linewidths=1)
            ax.annotate(label, (x, y), fontsize=9, fontweight="bold", color=color,
                        xytext=(6, 6), textcoords="offset points",
                        bbox=dict(fc="white", alpha=0.7, ec="none", pad=1))

    ax.axhline(0, color="#CCCCCC", lw=0.8, ls="--")
    ax.axvline(0, color="#CCCCCC", lw=0.8, ls="--")

    # Legend
    handles = [mpatches.Patch(color=c, label=l)
               for l, c in cat_type_colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=10,
              framealpha=0.9, edgecolor="#CCCCCC",
              title="Category Type", title_fontsize=11)

    ax.set_xlabel(f"Dimension 1 ({pct[0]:.1f}% inertia)", fontsize=13, fontfamily=EN_FONT)
    ax.set_ylabel(f"Dimension 2 ({pct[1]:.1f}% inertia)", fontsize=13, fontfamily=EN_FONT)
    ax.set_title("多重对应分析 (MCA) 双标图\nMultiple Correspondence Analysis Biplot",
                 fontsize=16, fontweight="bold", pad=15)

    # Dimension interpretation annotations
    dim_interp = mca["dimension_interpretations"]
    d1_neg = list(dim_interp["Dimension_1"]["negative_pole"].keys())[:2]
    d1_pos = list(dim_interp["Dimension_1"]["positive_pole"].keys())[:2]
    d2_neg = list(dim_interp["Dimension_2"]["negative_pole"].keys())[:2]
    d2_pos = list(dim_interp["Dimension_2"]["positive_pole"].keys())[:2]

    d1_neg_labels = [k.split("__")[1] if "__" in k else k for k in d1_neg]
    d1_pos_labels = [k.split("__")[1] if "__" in k else k for k in d1_pos]
    d2_neg_labels = [k.split("__")[1] if "__" in k else k for k in d2_neg]
    d2_pos_labels = [k.split("__")[1] if "__" in k else k for k in d2_pos]

    add_text_box(ax,
                 f"Dim1 解读:\n"
                 f"  (-) {', '.join(d1_neg_labels)}\n"
                 f"  (+) {', '.join(d1_pos_labels)}\n"
                 f"Dim2 解读:\n"
                 f"  (-) {', '.join(d2_neg_labels)}\n"
                 f"  (+) {', '.join(d2_pos_labels)}\n"
                 f"累计惯量: {mca['cumulative_inertia'][1]:.1f}%",
                 0.01, 0.45, fontsize=8.5,
                 boxcolor="#FFF3E0", edgecolor="#FF9800")

    add_source_note(fig, "F04", f"MCA on {mca['n_observations']} obs, {mca['n_components']} components")
    fig.savefig(FIG_DIR / "fig_f04_mca_biplot.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# F05  Geographic Inequality
# ============================================================================
def fig_f05():
    print("  [4/12] fig_f05_geographic_inequality ...")
    geo = causal["6_geographic_inequality"]
    metrics = ["innovation_depth", "iSTAR_level", "tech_generation"]
    metric_labels = ["创新深度\nInnovation Depth", "iSTAR协同层级\niSTAR Level",
                     "技术代际\nTech Generation"]
    bar_colors = ["#E63946", "#457B9D", "#2A9D8F"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, bar_colors)):
        ax = axes[idx]
        data = geo[metric]

        # Theil by province
        theil_prov = data["theil_by_province"]
        between_prov = theil_prov["pct_between"]
        within_prov = theil_prov["pct_within"]

        # Theil by region
        theil_reg = data["theil_by_region"]
        between_reg = theil_reg["pct_between"]
        within_reg = theil_reg["pct_within"]

        gini = data["overall_gini"]

        # stacked bars
        categories = ["Province\n省份", "Region\n区域"]
        between_vals = [between_prov, between_reg]
        within_vals = [within_prov, within_reg]

        bars1 = ax.barh(categories, within_vals, color=color, alpha=0.5,
                        label="Within 组内", edgecolor="white")
        bars2 = ax.barh(categories, between_vals, left=within_vals,
                        color=color, alpha=0.95, label="Between 组间",
                        edgecolor="white")

        # add percentage labels
        for bar, val in zip(bars1, within_vals):
            ax.text(val/2, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        for bar, val, left in zip(bars2, between_vals, within_vals):
            if val > 2:
                ax.text(left + val/2, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")

        ax.set_xlim(0, 105)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Theil Decomposition (%)", fontsize=10, fontfamily=EN_FONT)

        # Gini badge
        ax.text(85, 1.3, f"Gini\n{gini:.3f}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color,
                bbox=dict(fc="white", ec=color, boxstyle="round,pad=0.5",
                          lw=2))

        # top/bottom provinces
        gini_by_prov = data["gini_by_province"]
        prov_sorted = sorted(gini_by_prov.items(), key=lambda x: x[1], reverse=True)
        top3 = [f"{p}: {v:.3f}" for p, v in prov_sorted[:3]]
        ax.text(0.98, 0.02, "Top inequality:\n" + "\n".join(top3),
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                bbox=dict(fc="#FFF8E1", ec="#FFB300", alpha=0.9,
                          boxstyle="round,pad=0.4"))

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("区域教育资源不平等 Theil 分解\nGeographic Inequality: Theil Decomposition & Gini Coefficients",
                 fontsize=16, fontweight="bold", y=1.02)

    # global annotation
    fig.text(0.5, -0.04,
             "Insight: 组内不平等远大于组间不平等（>98%），表明省份/区域间差异微小，"
             "不平等主要源自省份内部的学校/案例差异。\n"
             "Within-group inequality dominates (>98%), indicating that disparity arises within provinces, "
             "not between them.",
             ha="center", fontsize=10, fontstyle="italic",
             bbox=dict(fc="#E3F2FD", ec="#1976D2", alpha=0.9, boxstyle="round,pad=0.5"),
             wrap=True)

    add_source_note(fig, "F05", "Theil decomposition, Gini coefficient")
    fig.savefig(FIG_DIR / "fig_f05_geographic_inequality.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# F06  Cluster ANOVA (eta-squared)
# ============================================================================
def fig_f06():
    print("  [5/12] fig_f06_cluster_anova ...")
    cluster = causal["7_cluster_profiling"]
    anova = cluster["anova_continuous"]
    chi_sq = cluster["chi_square_categorical"]

    # continuous variables
    items = []
    for var, vals in anova.items():
        items.append({
            "var": var,
            "eta2": vals["eta_squared"],
            "F": vals["F_statistic"],
            "p": vals.get("p_value", 0),
            "interp": vals["effect_size_interpretation"],
            "type": "continuous"
        })
    # categorical variables
    for var, vals in chi_sq.items():
        items.append({
            "var": var,
            "eta2": vals["cramers_v"] ** 2,   # approximate
            "F": vals["chi2_statistic"],
            "p": vals.get("p_value", 0),
            "interp": vals["effect_size_interpretation"],
            "type": "categorical"
        })

    items.sort(key=lambda x: x["eta2"], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    colors_map = {"large": "#E63946", "medium": "#F4A261", "small": "#90E0EF"}
    y_pos = range(len(items))
    for i, item in enumerate(items):
        color = colors_map.get(item["interp"], "#CCCCCC")
        bar = ax.barh(i, item["eta2"], color=color, edgecolor="white",
                      height=0.7, alpha=0.9)
        # significance star
        star = ""
        if item["p"] < 0.001:
            star = " ***"
        elif item["p"] < 0.01:
            star = " **"
        elif item["p"] < 0.05:
            star = " *"
        ax.text(item["eta2"] + 0.005, i,
                f"  {item['eta2']:.3f}{star}",
                va="center", fontsize=10, fontweight="bold",
                color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([it["var"] for it in items], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Effect Size (eta-squared / Cramer's V squared)", fontsize=12, fontfamily=EN_FONT)
    ax.set_title("聚类 ANOVA 效应量分析\nCluster Profiling: Effect Sizes (Eta-squared)",
                 fontsize=16, fontweight="bold", pad=15)

    # Effect size legend
    legend_elements = [
        mpatches.Patch(facecolor="#E63946", label="Large (>0.14)"),
        mpatches.Patch(facecolor="#F4A261", label="Medium (0.06-0.14)"),
        mpatches.Patch(facecolor="#90E0EF", label="Small (<0.06)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              title="Effect Size Category", title_fontsize=11,
              framealpha=0.9)

    # benchmark lines
    ax.axvline(0.01, color="#CCCCCC", ls=":", lw=1, alpha=0.6)
    ax.axvline(0.06, color="#F4A261", ls=":", lw=1, alpha=0.6)
    ax.axvline(0.14, color="#E63946", ls=":", lw=1, alpha=0.6)
    ax.text(0.01, -0.8, "Small", fontsize=7, color="#999999", ha="center")
    ax.text(0.06, -0.8, "Medium", fontsize=7, color="#F4A261", ha="center")
    ax.text(0.14, -0.8, "Large", fontsize=7, color="#E63946", ha="center")

    add_text_box(ax,
                 f"10个聚类, N={cluster['n_observations']:,}\n"
                 "D4_人机互信 eta^2=0.99 (极强区分力)\n"
                 "D3_循证教学, D1_深度学习 次之\n"
                 "数字教学法维度是聚类核心驱动力\n"
                 "*** p<0.001  ** p<0.01  * p<0.05",
                 0.55, 0.95, fontsize=9,
                 boxcolor="#FCE4EC", edgecolor="#E91E63")

    ax.set_xlim(0, max(it["eta2"] for it in items) * 1.15)
    add_source_note(fig, "F06", f"ANOVA on {cluster['n_clusters']} clusters")
    fig.savefig(FIG_DIR / "fig_f06_cluster_anova.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# B03  LLM Model Landscape Bubble Chart
# ============================================================================
def fig_b03():
    print("  [6/12] fig_b03_model_landscape ...")
    tool_counts = df["公司"].value_counts().head(20).reset_index()
    tool_counts.columns = ["company", "count"]

    # product category mode for each company
    cat_map = df.groupby("公司")["产品分类"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "未知")
    tool_counts["category"] = tool_counts["company"].map(cat_map).fillna("未知")

    # company type classification
    large_platform = ["豆包","文心一言","通义千问大模型","智谱清言","腾讯元宝","飞书AI"]
    open_source = ["DeepSeek 大模型"]
    def classify(name):
        if name in large_platform:
            return "大型平台\nLarge Platform"
        elif name in open_source:
            return "开源模型\nOpen Source"
        else:
            return "垂直厂商\nVertical Vendor"

    tool_counts["company_type"] = tool_counts["company"].apply(classify)

    # assign product category numeric
    cats = tool_counts["category"].unique()
    cat_to_num = {c: i for i, c in enumerate(cats)}
    tool_counts["cat_num"] = tool_counts["category"].map(cat_to_num)

    # rank
    tool_counts["rank"] = range(1, len(tool_counts) + 1)

    fig, ax = plt.subplots(figsize=(16, 11))

    type_colors = {
        "大型平台\nLarge Platform": "#E63946",
        "开源模型\nOpen Source": "#2A9D8F",
        "垂直厂商\nVertical Vendor": "#457B9D",
    }

    for _, row in tool_counts.iterrows():
        color = type_colors.get(row["company_type"], "#999999")
        ax.scatter(row["cat_num"], row["rank"],
                   s=row["count"] * 3, c=color, alpha=0.7,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(f"{row['company']}\n(n={row['count']})",
                    (row["cat_num"], row["rank"]),
                    fontsize=8, ha="center", va="center",
                    fontweight="bold",
                    color="#333333")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Market Share Rank (1=Top)", fontsize=12, fontfamily=EN_FONT)
    ax.set_xlabel("Product Category 产品分类", fontsize=12)
    ax.invert_yaxis()
    ax.set_title("LLM/AI 工具市场格局\nLLM Model Landscape: Market Position Bubble Chart",
                 fontsize=16, fontweight="bold", pad=15)

    # legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in type_colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=10,
              title="Company Type 企业类型", title_fontsize=11)

    # quadrant lines
    mid_x = len(cats) / 2 - 0.5
    mid_y = 10.5
    ax.axhline(mid_y, color="#CCCCCC", ls="--", lw=1, alpha=0.5)
    ax.axvline(mid_x, color="#CCCCCC", ls="--", lw=1, alpha=0.5)
    ax.text(0.02, 0.02, "High Share\nNiche Category", transform=ax.transAxes,
            fontsize=9, color="#999999", style="italic")
    ax.text(0.98, 0.02, "High Share\nBroad Category", transform=ax.transAxes,
            fontsize=9, color="#999999", style="italic", ha="right")

    add_text_box(ax,
                 "豆包(427次)与DeepSeek(300次)\n"
                 "主导教育AI市场\n"
                 "即梦AI(180)、剪映AI(78)紧随其后\n"
                 "大型平台占据绝对优势地位",
                 0.01, 0.65, fontsize=9,
                 boxcolor="#E8F5E9", edgecolor="#4CAF50")

    add_source_note(fig, "B03", "Top 20 companies by frequency")
    fig.savefig(FIG_DIR / "fig_b03_model_landscape.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# B04  Technology Co-occurrence Network
# ============================================================================
def fig_b04():
    print("  [7/12] fig_b04_cooccurrence_network ...")
    # Parse tech keywords
    tech_lists = []
    for raw in df["技术要素"].dropna():
        try:
            items = ast.literal_eval(raw)
            if isinstance(items, list):
                cleaned = [s.strip().strip('"').strip("'") for s in items if len(s.strip()) > 1]
                tech_lists.append(cleaned)
        except:
            pass

    # Count individual keywords
    all_kw = Counter()
    for lst in tech_lists:
        for kw in lst:
            all_kw[kw] += 1

    top_kw = [kw for kw, _ in all_kw.most_common(35)]
    top_set = set(top_kw)

    # Co-occurrence
    cooc = Counter()
    for lst in tech_lists:
        filtered = [kw for kw in lst if kw in top_set]
        for a, b in combinations(sorted(set(filtered)), 2):
            cooc[(a, b)] += 1

    # Build graph
    G = nx.Graph()
    for kw in top_kw:
        G.add_node(kw, freq=all_kw[kw])

    for (a, b), w in cooc.items():
        if w >= 5:
            G.add_edge(a, b, weight=w)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes) < 3:
        print("    [WARN] Not enough nodes for network, skipping ...")
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.5, 0.5, "Insufficient co-occurrence data for network",
                ha="center", va="center", fontsize=16)
        fig.savefig(FIG_DIR / "fig_b04_cooccurrence_network.png", dpi=DPI)
        plt.close(fig)
        return

    # Community detection
    try:
        communities = louvain_communities(G, seed=42)
    except:
        communities = [set(G.nodes)]

    # Assign community colors
    comm_colors = PAL_VIVID[:len(communities)]
    node_color_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_color_map[node] = comm_colors[idx % len(comm_colors)]

    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=80)

    # edges
    edges = G.edges(data=True)
    max_w = max(d["weight"] for _, _, d in edges) if edges else 1
    for u, v, d in edges:
        w = d["weight"]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#CCCCCC", alpha=min(0.15 + 0.6 * w / max_w, 0.8),
                linewidth=0.5 + 3 * w / max_w, zorder=1)

    # nodes
    node_list = list(G.nodes)
    node_sizes = [G.nodes[n].get("freq", 10) * 4 for n in node_list]
    node_colors = [node_color_map.get(n, "#999999") for n in node_list]

    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                           node_size=node_sizes, node_color=node_colors,
                           alpha=0.85, edgecolors="white", linewidths=1.5,
                           ax=ax)

    # labels
    for node in node_list:
        x, y = pos[node]
        freq = G.nodes[node].get("freq", 0)
        fsize = 7 + min(freq / 80, 5)
        ax.text(x, y, node, fontsize=fsize, ha="center", va="center",
                fontweight="bold", color="#222222",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    ax.set_title("技术关键词共现网络 (Louvain社区检测)\nTechnology Keyword Co-occurrence Network",
                 fontsize=16, fontweight="bold", pad=15)
    ax.axis("off")

    # community legend
    legend_elements = [mpatches.Patch(facecolor=comm_colors[i],
                                      label=f"Community {i+1} ({len(c)} keywords)")
                       for i, c in enumerate(communities)]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              title="Communities", title_fontsize=10, framealpha=0.9)

    add_text_box(ax,
                 f"节点={len(G.nodes)}, 边={len(G.edges)}\n"
                 f"社区数={len(communities)}\n"
                 "节点大小=出现频率\n"
                 "边粗细=共现次数\n"
                 "Louvain算法自动识别技术簇群",
                 0.75, 0.98, fontsize=9,
                 boxcolor="#F3E5F5", edgecolor="#9C27B0")

    add_source_note(fig, "B04", "Co-occurrence network, Louvain community detection")
    fig.savefig(FIG_DIR / "fig_b04_cooccurrence_network.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# D02  Ecosystem Treemap
# ============================================================================
def fig_d02():
    print("  [8/12] fig_d02_ecosystem_treemap ...")
    import squarify

    company_counts = df["公司"].value_counts().head(30)
    # map to product category
    cat_mode = df.groupby("公司")["产品分类"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "其他")

    labels = []
    sizes = []
    colors = []

    cat_list = cat_mode[company_counts.index].unique()
    cat_palette = {c: PAL_VIVID[i % len(PAL_VIVID)] for i, c in enumerate(cat_list)}

    for company, count in company_counts.items():
        cat = cat_mode.get(company, "其他")
        labels.append(f"{company}\n{count}")
        sizes.append(count)
        colors.append(cat_palette.get(cat, "#999999"))

    fig, ax = plt.subplots(figsize=(16, 11))

    rects = squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85,
                          text_kwargs={"fontsize": 8, "fontweight": "bold",
                                       "color": "white",
                                       "fontfamily": CN_FONT},
                          ax=ax, bar_kwargs=dict(linewidth=2, edgecolor="white"))

    ax.set_title("企业生态系统 Treemap (Top 30)\nCompany Ecosystem Treemap by Market Presence",
                 fontsize=16, fontweight="bold", pad=15)
    ax.axis("off")

    # Category legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in cat_palette.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Product Category 产品分类", title_fontsize=9,
              ncol=2, framealpha=0.9)

    add_text_box(ax,
                 "面积=产品出现频次\n"
                 "颜色=产品分类\n"
                 "豆包(427)占据最大份额\n"
                 "DeepSeek(300)紧随其后\n"
                 "市场集中度高: CR2 > 40%",
                 0.01, 0.25, fontsize=9,
                 boxcolor="#E0F7FA", edgecolor="#00ACC1")

    add_source_note(fig, "D02", "Top 30 companies by frequency")
    fig.savefig(FIG_DIR / "fig_d02_ecosystem_treemap.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# E04  Digital Pedagogy 4D Radar
# ============================================================================
def fig_e04():
    print("  [9/12] fig_e04_digital_pedagogy_radar ...")
    scenarios = ["助学", "助教", "助评", "助管"]
    dims = ["D1_深度学习", "D2_绿色鲁棒", "D3_循证教学", "D4_人机互信"]
    dim_labels = ["D1 深度学习\nDeep Learning", "D2 绿色鲁棒\nGreen & Robust",
                  "D3 循证教学\nEvidence-Based", "D4 人机互信\nHuman-AI Trust"]

    sub = df[df["应用场景（一级）"].isin(scenarios)]

    means = {}
    for scen in scenarios:
        s = sub[sub["应用场景（一级）"] == scen]
        means[scen] = [s[d].mean() for d in dims]

    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    scen_colors = {"助学":"#E63946","助教":"#457B9D","助评":"#2A9D8F","助管":"#E9C46A"}

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

    for scen in scenarios:
        values = means[scen] + [means[scen][0]]
        color = scen_colors[scen]
        ax.plot(angles, values, "-o", linewidth=2.5, label=scen, color=color,
                markersize=8, zorder=3)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666666")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11,
              title="Application Scenario", title_fontsize=12)

    ax.set_title("数字教学法4维雷达图 (按应用场景)\nDigital Pedagogy 4D Radar by Scenario",
                 fontsize=16, fontweight="bold", y=1.12)

    # Mean scores text
    center_text = "Mean Scores:\n"
    for scen in scenarios:
        avg = np.mean(means[scen])
        center_text += f"  {scen}: {avg:.3f}\n"
    fig.text(0.82, 0.25, center_text,
             fontsize=10, fontfamily=CN_FONT,
             bbox=dict(fc="#FFF8E1", ec="#FFC107", boxstyle="round,pad=0.5",
                       alpha=0.92))

    add_text_box(ax,
                 "助学场景: D1(深度学习)得分最高\n"
                 "助教场景: D3(循证教学)突出\n"
                 "助评/助管: D2,D4提升空间大\n"
                 "整体D2(绿色鲁棒)最为薄弱",
                 -0.15, -0.15, fontsize=9, transform=ax.transAxes,
                 boxcolor="#E8F5E9", edgecolor="#4CAF50")

    add_source_note(fig, "E04", "4D digital pedagogy framework by scenario")
    fig.savefig(FIG_DIR / "fig_e04_digital_pedagogy_radar.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# E05  TechGen x Scenario Bubble
# ============================================================================
def fig_e05():
    print("  [10/12] fig_e05_techgen_scenario_bubble ...")
    sub = df.dropna(subset=["产品技术代际", "应用场景（一级）", "创新深度评分"])
    ct = sub.groupby(["产品技术代际", "应用场景（一级）"]).agg(
        count=("创新深度评分", "size"),
        mean_innovation=("创新深度评分", "mean")
    ).reset_index()

    # Tech generation order
    gen_order = ["Gen1_传统信息化","Gen2_互联网+","Gen3_AI辅助","Gen4_大模型","Gen5_多模态AI"]
    gen_to_num = {g: i for i, g in enumerate(gen_order)}
    scen_order = ["助学","助教","助评","助管","助育","助研","未提及"]
    scen_to_num = {s: i for i, s in enumerate(scen_order)}

    ct["x"] = ct["产品技术代际"].map(gen_to_num)
    ct["y"] = ct["应用场景（一级）"].map(scen_to_num)
    ct = ct.dropna(subset=["x", "y"])

    fig, ax = plt.subplots(figsize=(16, 11))

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=ct["mean_innovation"].min(),
                         vmax=ct["mean_innovation"].max())

    scatter = ax.scatter(ct["x"], ct["y"],
                        s=ct["count"] * 3,
                        c=ct["mean_innovation"],
                        cmap=cmap, alpha=0.8,
                        edgecolors="white", linewidths=1.5, zorder=3)

    # Add count labels
    for _, row in ct.iterrows():
        if row["count"] > 15:
            ax.text(row["x"], row["y"], f'{int(row["count"])}',
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if row["mean_innovation"] > 3 else "#333333")

    ax.set_xticks(range(len(gen_order)))
    ax.set_xticklabels([g.replace("_", "\n") for g in gen_order], fontsize=10)
    ax.set_yticks(range(len(scen_order)))
    ax.set_yticklabels(scen_order, fontsize=11)

    ax.set_xlabel("产品技术代际 Technology Generation", fontsize=12)
    ax.set_ylabel("应用场景 Application Scenario", fontsize=12)
    ax.set_title("技术代际 x 应用场景 气泡矩阵\nTechnology Generation x Scenario Bubble Matrix",
                 fontsize=16, fontweight="bold", pad=15)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean Innovation Depth 平均创新深度", fontsize=11)

    # quadrant annotations
    ax.axvline(2, color="#CCCCCC", ls="--", lw=1, alpha=0.5)
    ax.text(0.5, -0.8, "Traditional\n传统技术", ha="center", fontsize=9,
            color="#999999", style="italic")
    ax.text(3.5, -0.8, "Frontier AI\n前沿AI", ha="center", fontsize=9,
            color="#999999", style="italic")

    add_text_box(ax,
                 "气泡大小=案例数量\n"
                 "颜色=平均创新深度评分\n"
                 "Gen4大模型 x 助学 = 最大集群\n"
                 "前沿AI代际创新深度更高(绿色)",
                 0.01, 0.99, fontsize=9,
                 boxcolor="#FFF3E0", edgecolor="#FF9800")

    # size legend
    for s_val in [50, 200, 500]:
        ax.scatter([], [], s=s_val * 3, c="#CCCCCC", alpha=0.6,
                   edgecolors="#999999", label=f"n={s_val}")
    ax.legend(loc="upper right", fontsize=9, title="Count", title_fontsize=10,
              framealpha=0.9, scatterpoints=1)

    add_source_note(fig, "E05", "Tech generation x scenario matrix")
    fig.savefig(FIG_DIR / "fig_e05_techgen_scenario_bubble.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# A05  Five-Education Radar
# ============================================================================
def fig_a05():
    print("  [11/12] fig_a05_five_edu_radar ...")
    # Map cultivation direction to five-education
    wu_yu = {"智育": 0, "德育": 0, "美育": 0, "体育": 0, "劳育": 0}

    for val in df["培养方向"].dropna():
        val_str = str(val)
        if "智" in val_str:
            wu_yu["智育"] += 1
        if "德" in val_str:
            wu_yu["德育"] += 1
        if "美" in val_str:
            wu_yu["美育"] += 1
        if "体" in val_str:
            wu_yu["体育"] += 1
        if "劳" in val_str:
            wu_yu["劳育"] += 1

    total = sum(wu_yu.values())
    actual = {k: v / total for k, v in wu_yu.items()}
    ideal = {k: 0.2 for k in wu_yu}

    categories = list(wu_yu.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    actual_vals = [actual[c] for c in categories] + [actual[categories[0]]]
    ideal_vals = [ideal[c] for c in categories] + [ideal[categories[0]]]

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(projection="polar"))

    # Ideal (reference)
    ax.plot(angles, ideal_vals, "--", linewidth=2, color="#AAAAAA", label="理想均衡 Ideal Balance")
    ax.fill(angles, ideal_vals, alpha=0.05, color="#999999")

    # Actual
    ax.plot(angles, actual_vals, "-o", linewidth=3, color="#E63946",
            label="实际分布 Actual", markersize=10, zorder=3)
    ax.fill(angles, actual_vals, alpha=0.15, color="#E63946")

    # Gap visualization
    for i in range(N):
        gap = actual_vals[i] - ideal_vals[i]
        color = "#4CAF50" if gap >= 0 else "#FF5722"
        ax.annotate(f"{gap:+.1%}",
                    xy=(angles[i], max(actual_vals[i], ideal_vals[i]) + 0.02),
                    fontsize=12, fontweight="bold", ha="center", color=color,
                    bbox=dict(fc="white", ec=color, alpha=0.8, boxstyle="round,pad=0.2"))

    cat_labels = [f"{c}\n({wu_yu[c]:,})" for c in categories]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(actual_vals) * 1.2)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

    ax.set_title("五育 (Five-Education) 多层雷达图\nFive-Education Radar: Actual vs Ideal Balance",
                 fontsize=16, fontweight="bold", y=1.12)

    # Gap analysis box
    max_gap_cat = max(wu_yu, key=wu_yu.get)
    min_gap_cat = min(wu_yu, key=wu_yu.get)
    fig.text(0.82, 0.2,
             f"Gap Analysis:\n"
             f"最强: {max_gap_cat} ({actual[max_gap_cat]:.1%})\n"
             f"最弱: {min_gap_cat} ({actual[min_gap_cat]:.1%})\n"
             f"不均衡度: {max(actual.values())/max(min(actual.values()),0.001):.1f}x\n\n"
             f"智育严重偏重, 劳育/体育亟需加强\n"
             f"AI教育产品'五育并举'任重道远",
             fontsize=10, fontfamily=CN_FONT,
             bbox=dict(fc="#FCE4EC", ec="#E91E63", boxstyle="round,pad=0.6",
                       alpha=0.92))

    add_source_note(fig, "A05", "Five-Education (Wu-Yu) distribution analysis")
    fig.savefig(FIG_DIR / "fig_a05_five_edu_radar.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# E06  Dashboard (3x2)
# ============================================================================
def fig_e06():
    print("  [12/12] fig_e06_dashboard ...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    theme_colors = PAL_VIVID

    # --- Panel 1: 三赋能 pie ---
    ax1 = fig.add_subplot(gs[0, 0])
    san_counts = df["三赋能分类"].value_counts()
    pie_colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
    wedges, texts, autotexts = ax1.pie(
        san_counts.values, labels=san_counts.index,
        autopct="%1.1f%%", startangle=90,
        colors=pie_colors[:len(san_counts)],
        textprops={"fontsize": 10},
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax1.set_title("Panel 1: 三赋能分布\nTri-Empowerment", fontsize=12, fontweight="bold")

    # --- Panel 2: iSTAR bar ---
    ax2 = fig.add_subplot(gs[0, 1])
    istar_counts = df["iSTAR人机协同层级"].value_counts().sort_index()
    istar_colors = ["#03045E", "#0077B6", "#00B4D8"]
    bars = ax2.bar(range(len(istar_counts)), istar_counts.values,
                   color=istar_colors[:len(istar_counts)],
                   edgecolor="white", linewidth=1.5)
    ax2.set_xticks(range(len(istar_counts)))
    ax2.set_xticklabels(istar_counts.index, fontsize=9)
    for bar, val in zip(bars, istar_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{val:,}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Panel 2: iSTAR 人机协同层级\niSTAR Collaboration Level", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count")

    # --- Panel 3: Innovation depth histogram ---
    ax3 = fig.add_subplot(gs[1, 0])
    inno = df["创新深度评分"].dropna()
    hist_colors = ["#FF006E", "#8338EC", "#3A86FF", "#FB5607", "#FFBE0B"]
    counts, bins, patches = ax3.hist(inno, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                                     edgecolor="white", linewidth=1.5)
    for patch, color in zip(patches, hist_colors):
        patch.set_facecolor(color)
    for i, (c, b) in enumerate(zip(counts, bins)):
        ax3.text(b + 0.5, c + 20, f"{int(c):,}", ha="center",
                 fontsize=9, fontweight="bold")
    ax3.set_xlabel("Innovation Depth Score 创新深度评分", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Panel 3: 创新深度评分分布\nInnovation Depth Distribution", fontsize=12, fontweight="bold")

    # --- Panel 4: Tech generation donut ---
    ax4 = fig.add_subplot(gs[1, 1])
    gen_counts = df["产品技术代际"].value_counts()
    gen_order = ["Gen1_传统信息化","Gen2_互联网+","Gen3_AI辅助","Gen4_大模型","Gen5_多模态AI"]
    gen_counts_ordered = gen_counts.reindex([g for g in gen_order if g in gen_counts.index])
    gen_colors = ["#264653","#2A9D8F","#E9C46A","#F4A261","#E63946"]
    wedges, texts, autotexts = ax4.pie(
        gen_counts_ordered.values,
        labels=[g.split("_")[1] for g in gen_counts_ordered.index],
        autopct="%1.1f%%", startangle=140,
        colors=gen_colors[:len(gen_counts_ordered)],
        textprops={"fontsize": 9},
        pctdistance=0.8,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax4.set_title("Panel 4: 技术代际\nTech Generation Donut", fontsize=12, fontweight="bold")

    # --- Panel 5: 智慧教育境界 stacked bar by scenario ---
    ax5 = fig.add_subplot(gs[2, 0])
    cross = pd.crosstab(df["应用场景（一级）"], df["智慧教育境界"], normalize="index")
    jj_order = ["第一境界_智慧环境","第二境界_教学模式","第三境界_制度变革"]
    cols_present = [c for c in jj_order if c in cross.columns]
    cross = cross[cols_present]
    jj_colors = ["#48BFE3", "#7400B8", "#FF006E"]
    cross.plot.barh(stacked=True, ax=ax5, color=jj_colors[:len(cols_present)],
                    edgecolor="white", linewidth=0.8)
    ax5.set_xlabel("Proportion", fontsize=10)
    ax5.set_title("Panel 5: 智慧教育境界 (按场景)\nSmart Education Level by Scenario", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=7, loc="lower right")

    # --- Panel 6: Key metrics text ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    metrics_text = (
        f"========= Key Metrics =========\n\n"
        f"Total Records:  {len(df):,}\n"
        f"Unique Companies:  {df['公司'].nunique()}\n"
        f"Unique Tools:  {df['工具标准名'].nunique()}\n"
        f"Provinces Covered:  {df['省份'].nunique()}\n\n"
        f"Mean Innovation Depth:  {df['创新深度评分'].mean():.2f}\n"
        f"Median Innovation:  {df['创新深度评分'].median():.1f}\n\n"
        f"Top Company: {df['公司'].value_counts().index[0]}  "
        f"({df['公司'].value_counts().iloc[0]:,})\n"
        f"Top Scenario: {df['应用场景（一级）'].value_counts().index[0]}  "
        f"({df['应用场景（一级）'].value_counts().iloc[0]:,})\n\n"
        f"Ordinal Logistic R2:  {causal['3_ordinal_logistic']['pseudo_r_squared']:.3f}\n"
        f"RF Accuracy (CV):  {causal['10_predictive_modeling']['cross_validation']['accuracy_mean']:.3f}\n"
        f"SEM Full Model R2:  {causal['5_SEM_path_analysis']['full_model_R2']:.3f}\n"
    )
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
             fontsize=11, fontfamily=EN_FONT, va="top",
             bbox=dict(fc="#F5F5F5", ec="#616161", boxstyle="round,pad=0.8",
                       alpha=0.95, lw=1.5),
             linespacing=1.5)
    ax6.set_title("Panel 6: 关键指标\nKey Statistics", fontsize=12, fontweight="bold")

    fig.suptitle("AI教育产品综合仪表盘\nComprehensive AI-Education Dashboard",
                 fontsize=18, fontweight="bold", y=1.01)

    add_source_note(fig, "E06", "Multi-panel comprehensive dashboard")
    fig.savefig(FIG_DIR / "fig_e06_dashboard.png", dpi=DPI)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  viz_part2.py  -  Generating 12 Advanced Figures")
    print("=" * 60)

    fig_f02()
    fig_f03()
    fig_f04()
    fig_f05()
    fig_f06()
    fig_b03()
    fig_b04()
    fig_d02()
    fig_e04()
    fig_e05()
    fig_a05()
    fig_e06()

    print("=" * 60)
    print(f"  All 12 figures saved to: {FIG_DIR}")
    print("=" * 60)
