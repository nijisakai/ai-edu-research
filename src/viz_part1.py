#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_part1.py  –  AI-Education Research · Publication-Quality Figures (Part 1)
=============================================================================
Generates 12 high-impact figures from the annotated dataset.
Output: /Users/sakai/Desktop/产业调研/ai-edu-research/output/figures/
"""

import warnings, json, textwrap, colorsys, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm, ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import squarify

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA   = "/Users/sakai/Desktop/产业调研/ai-edu-research/output/教育产品统计_V6_框架标注.csv"
CAUSAL = "/Users/sakai/Desktop/产业调研/ai-edu-research/output/causal_analysis_results.json"
OUTDIR = pathlib.Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
with open(CAUSAL, "r") as f:
    causal = json.load(f)

# ─── Global Style ──────────────────────────────────────────────────────────
CN_FONT = "Source Han Sans CN"
EN_FONT = "Helvetica Neue"
DPI     = 400

# Nature / Science journal palette (8 colors)
PAL8 = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC9A6C"]
# Extended 12-color palette
PAL12 = PAL8 + ["#7E6148", "#B09C85", "#D65DB1", "#845EC2"]

plt.rcParams.update({
    "font.family":          CN_FONT,
    "font.sans-serif":      [CN_FONT, EN_FONT, "Arial"],
    "axes.unicode_minus":   False,
    "savefig.dpi":          DPI,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.3,
    "figure.facecolor":     "white",
    "axes.facecolor":       "#FAFAFA",
    "axes.edgecolor":       "#CCCCCC",
    "axes.grid":            False,
})

# ─── Helpers ───────────────────────────────────────────────────────────────
def add_fig_number(ax, num, title_cn, title_en,
                   source="数据来源：全国AI教育应用案例数据库 (N={:,})".format(3815)):
    """Add bilingual title on top, figure number + source at bottom."""
    ax.set_title(f"{title_cn}\n", fontsize=16, fontweight="bold",
                 fontfamily=CN_FONT, pad=18)
    ax.text(0.5, 1.03, title_en, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, fontfamily=EN_FONT,
            color="#555555", style="italic")
    ax.text(0.0, -0.08, f"Fig. {num}",
            transform=ax.transAxes, fontsize=8, fontfamily=EN_FONT,
            fontweight="bold", color="#333333")
    ax.text(1.0, -0.08, source,
            transform=ax.transAxes, fontsize=7, fontfamily=CN_FONT,
            ha="right", color="#888888")

def add_fig_number_fig(fig, num, title_cn, title_en,
                       source="数据来源：全国AI教育应用案例数据库 (N={:,})".format(3815)):
    """Same but on figure level (for multi-axes figures)."""
    fig.suptitle(f"{title_cn}", fontsize=17, fontweight="bold",
                 fontfamily=CN_FONT, y=0.97)
    fig.text(0.5, 0.935, title_en, ha="center", fontsize=10,
             fontfamily=EN_FONT, color="#555555", style="italic")
    fig.text(0.02, 0.01, f"Fig. {num}", fontsize=8, fontfamily=EN_FONT,
             fontweight="bold", color="#333333")
    fig.text(0.98, 0.01, source, ha="right", fontsize=7,
             fontfamily=CN_FONT, color="#888888")

def callout(ax, text, xy, xytext, fontsize=9, color="#E64B35"):
    """Annotation callout box with arrow."""
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=fontsize, fontfamily=CN_FONT, color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="none", alpha=0.92),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                        connectionstyle="arc3,rad=0.15"),
        ha="center", va="center",
    )

def lighten(hex_color, amount=0.35):
    """Lighten a hex colour."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


# ═══════════════════════════════════════════════════════════════════════════
# FIG A01 — Province Map (simplified rectangle cartogram)
# ═══════════════════════════════════════════════════════════════════════════
def fig_a01():
    print("  [A01] Province map …")
    # Province approximate grid positions (row, col) for a cartogram
    # Each province mapped to approximate geographic position
    province_grid = {
        "黑龙江省": (0, 6), "吉林省": (1, 6), "辽宁省": (2, 6),
        "内蒙古自治区": (1, 4), "新疆维吾尔自治区": (2, 0),
        "西藏自治区": (4, 0), "青海省": (3, 1), "甘肃省": (3, 2),
        "宁夏回族自治区": (2, 3), "陕西省": (3, 3),
        "山西省": (2, 4), "河北省": (2, 5), "北京市": (1, 5),
        "天津市": (1.5, 5.5), "山东省": (3, 5),
        "河南省": (3, 4), "江苏省": (4, 5), "安徽省": (4, 4),
        "上海市": (4, 6), "浙江省": (5, 5), "江西省": (5, 4),
        "湖北省": (4, 3), "湖南省": (5, 3),
        "福建省": (5.5, 5.5), "台湾省": (6, 6),
        "广东省": (6, 4), "广西壮族自治区": (6, 3),
        "海南省": (7, 4), "四川省": (4, 2), "重庆市": (4.5, 2.5),
        "贵州省": (5, 2), "云南省": (6, 1),
    }
    prov_counts = df["省份"].value_counts()
    # Warm-to-hot colormap
    cmap = LinearSegmentedColormap.from_list("warm_hot",
        ["#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
         "#FC8D59", "#EF6548", "#D7301F", "#990000"])

    vmax = prov_counts.max()
    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-0.8, 8.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    for prov, (r, c) in province_grid.items():
        cnt = prov_counts.get(prov, 0)
        color = cmap(norm(cnt))
        rect = FancyBboxPatch((c - 0.42, r - 0.42), 0.84, 0.84,
                              boxstyle="round,pad=0.04",
                              facecolor=color, edgecolor="white", linewidth=1.2)
        ax.add_patch(rect)
        short = prov.replace("省", "").replace("市", "").replace("自治区", "")\
                     .replace("壮族", "").replace("回族", "").replace("维吾尔", "")
        if len(short) > 3:
            short = short[:3]
        ax.text(c, r - 0.08, short, ha="center", va="center",
                fontsize=7, fontfamily=CN_FONT, fontweight="bold",
                color="white" if cnt > vmax * 0.35 else "#333333")
        if cnt > 0:
            ax.text(c, r + 0.22, str(cnt), ha="center", va="center",
                    fontsize=5.5, fontfamily=EN_FONT,
                    color="white" if cnt > vmax * 0.35 else "#666666")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label("案例数量 / Case Count", fontsize=9, fontfamily=CN_FONT)

    # Top-5 callouts
    top5 = prov_counts.head(5)
    callout_offsets = [(2.5, -0.7), (-1.5, -0.6), (-2.5, 0.4), (2.5, -0.5), (-2, 0.5)]
    for i, (prov, cnt) in enumerate(top5.items()):
        if prov in province_grid:
            r, c = province_grid[prov]
            dx, dy = callout_offsets[i]
            callout(ax, f"{prov}: {cnt}例",
                    xy=(c, r), xytext=(c + dx, r + dy),
                    color=PAL8[i % len(PAL8)])

    # Insight box
    ax.text(0.5, 8.0, "   浙江省以1,005例领跑全国，占比26.3%；东部沿海地区集中度显著   ",
            ha="center", va="center", fontsize=10, fontfamily=CN_FONT,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#3C5488", ec="none", alpha=0.9))

    add_fig_number(ax, "A01", "AI教育应用案例省域分布热力地图",
                   "Geographic Distribution of AI-Education Cases across China")
    fig.savefig(OUTDIR / "fig_a01_province_map.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG A02 — Waffle Chart for School Stage
# ═══════════════════════════════════════════════════════════════════════════
def fig_a02():
    print("  [A02] Stage waffle …")
    stage_map = {
        "小学": "小学", "初中": "初中", "高中": "高中", "幼儿园": "幼儿园",
        "中学": "中学(其他)", "中小学": "中学(其他)", "小学/初中": "中学(其他)",
        "初中/高中": "中学(其他)", "中职": "中学(其他)",
    }
    stages = df["学段"].map(lambda x: stage_map.get(x, "其他"))
    counts = stages.value_counts()
    total = counts.sum()
    # Compute squares per category (100 total)
    N = 100
    raw_pcts = (counts / total * N)
    squares = raw_pcts.astype(int)
    # Distribute remainder
    remainder = N - squares.sum()
    frac = raw_pcts - squares
    for idx in frac.nlargest(remainder).index:
        squares[idx] += 1

    colors_map = {
        "小学": PAL8[0], "初中": PAL8[1], "高中": PAL8[2],
        "幼儿园": PAL8[3], "中学(其他)": PAL8[4], "其他": PAL8[5],
    }

    grid = []
    for cat in squares.index:
        grid.extend([cat] * squares[cat])

    fig, ax = plt.subplots(figsize=(14, 10))
    cols, rows = 10, 10
    gap = 0.12
    sz = 0.85
    for i, cat in enumerate(grid):
        r = i // cols
        c = i % cols
        rect = FancyBboxPatch(
            (c * (sz + gap), (rows - 1 - r) * (sz + gap)),
            sz, sz,
            boxstyle="round,pad=0.08",
            facecolor=colors_map.get(cat, "#AAAAAA"),
            edgecolor="white", linewidth=0.8, alpha=0.92
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.3, cols * (sz + gap))
    ax.set_ylim(-0.3, rows * (sz + gap) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend + percentages
    legend_y = rows * (sz + gap) + 1.2
    for i, (cat, n) in enumerate(squares.items()):
        pct = counts[cat] / total * 100
        handle = mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.05",
            facecolor=colors_map.get(cat, "#AAAAAA"))
        ax.text(cols * (sz + gap) + 0.8, (rows - 1 - i * 1.5) * (sz + gap),
                f"{cat}  {pct:.1f}%  ({counts[cat]:,}例)",
                fontsize=11, fontfamily=CN_FONT, fontweight="bold",
                color=colors_map.get(cat, "#333333"), va="center")

    # Each square = 1% note
    ax.text(0, -0.8, "每格 = 1% of total (N={:,})".format(total),
            fontsize=9, fontfamily=CN_FONT, color="#888888")

    # Insight
    ax.text(cols * (sz + gap) / 2, -1.6,
            "   小学阶段占比51.5%，AI教育应用集中于基础教育前端   ",
            ha="center", fontsize=10, fontfamily=CN_FONT, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#E64B35", ec="none", alpha=0.9))

    add_fig_number(ax, "A02", "学段分布华夫图",
                   "Waffle Chart of School Stage Distribution (1 square = 1%)")
    fig.savefig(OUTDIR / "fig_a02_stage_waffle.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG A03 — Lollipop Chart for Top 20 Subjects
# ═══════════════════════════════════════════════════════════════════════════
def fig_a03():
    print("  [A03] Subject lollipop …")
    subj = df[df["学科"] != "未提及"]["学科"].value_counts().head(20).sort_values()

    STEM = {"数学", "科学", "物理", "化学", "生物", "信息科技", "信息技术", "人工智能"}
    ARTS = {"美术", "音乐", "艺术"}

    cmap_lollipop = LinearSegmentedColormap.from_list(
        "cool_warm", ["#4DBBD5", "#00A087", "#F39B7F", "#E64B35"])
    norm = Normalize(0, len(subj) - 1)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, (name, val) in enumerate(subj.items()):
        color = cmap_lollipop(norm(i))
        ax.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=2.2, alpha=0.85)
        # Marker shape by group
        if name in STEM:
            marker = "D"
            label_tag = "STEM"
        elif name in ARTS:
            marker = "s"
            label_tag = "Arts"
        else:
            marker = "o"
            label_tag = "Hum."
        ax.plot(val, i, marker=marker, markersize=10, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.text(val + 12, i, f"{val}",
                va="center", fontsize=8, fontfamily=EN_FONT, color="#555555")

    ax.set_yticks(range(len(subj)))
    ax.set_yticklabels(subj.index, fontsize=9, fontfamily=CN_FONT)
    ax.set_xlabel("案例数量 / Case Count", fontsize=10, fontfamily=CN_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group legend
    for marker, label, c in [("D", "STEM", PAL8[0]),
                              ("s", "Arts", PAL8[3]),
                              ("o", "Humanities/Other", PAL8[1])]:
        ax.plot([], [], marker=marker, linestyle="none", markersize=8,
                color=c, label=label, markeredgecolor="white")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Insight callout
    callout(ax, "语文+数学占Top-2\n合计1,043例", xy=(subj.iloc[-1], len(subj)-1),
            xytext=(subj.iloc[-1] - 200, len(subj) - 4), color="#3C5488")

    add_fig_number(ax, "A03", "Top-20学科AI应用案例棒棒糖图",
                   "Lollipop Chart of Top 20 Subjects in AI-Education Cases")
    fig.savefig(OUTDIR / "fig_a03_subject_lollipop.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG A04 — Treemap for Scenario Distribution
# ═══════════════════════════════════════════════════════════════════════════
def fig_a04():
    print("  [A04] Scenario treemap …")
    # Build nested data: 一级 → 二级
    grp = df.groupby(["应用场景（一级）", "应用场景（二级）"]).size().reset_index(name="n")
    grp = grp.sort_values("n", ascending=False)

    # Color by 一级
    l1_cats = grp["应用场景（一级）"].unique()
    l1_colors = {c: PAL8[i % len(PAL8)] for i, c in enumerate(l1_cats)}

    sizes = grp["n"].values
    labels = []
    colors = []
    for _, row in grp.iterrows():
        l2 = row["应用场景（二级）"]
        if len(l2) > 8:
            l2 = l2[:8] + "…"
        labels.append(f"{row['应用场景（一级）']}\n{l2}\n({row['n']})")
        base = l1_colors[row["应用场景（一级）"]]
        # Slightly vary lightness for each sub-category
        colors.append(lighten(base, np.random.uniform(0, 0.3)))

    fig, ax = plt.subplots(figsize=(14, 10))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.88,
                  text_kwargs=dict(fontsize=6, fontfamily=CN_FONT,
                                   fontweight="bold", color="white",
                                   wrap=True),
                  ax=ax, pad=True)
    ax.axis("off")

    # Insight
    ax.text(0.5, -0.02,
            '   "助学-智能辅导系统"占全部场景的59.1%，平台应用高度集中   ',
            transform=ax.transAxes, ha="center", fontsize=10,
            fontfamily=CN_FONT, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#00A087", ec="none", alpha=0.9))

    add_fig_number(ax, "A04", "应用场景树状图 (一级→二级)",
                   "Treemap of Application Scenarios (L1 → L2 Nesting)")
    fig.savefig(OUTDIR / "fig_a04_scenario_treemap.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG B01 — Top 20 Tools Horizontal Bar
# ═══════════════════════════════════════════════════════════════════════════
def fig_b01():
    print("  [B01] Top 20 tools …")
    tool_df = df[df["工具标准名"].notna() & (df["工具标准名"] != "未提及")]
    tool_counts = tool_df["工具标准名"].value_counts().head(20).sort_values()
    total = tool_df["工具标准名"].value_counts().sum()

    # Map tool to type
    LLM = {"豆包", "DeepSeek 大模型", "文心一言", "Kimi", "通义千问大模型",
            "星火大模型", "智谱清言", "腾讯元宝", "豆包智能体", "ChatGPT"}
    IMG = {"即梦AI", "即梦 AI", "剪映AI"}
    PLAT = {"希沃白板", "国家智慧教育平台", "希沃AI", "班级优化大师",
            "钉钉AI", "希沃易课堂", "智学网", "飞书AI", "央馆领航AI",
            "文曲智阅", "扣子AI", "问卷星", "数字人"}

    def tool_type(t):
        if t in LLM: return "LLM 大模型"
        if t in IMG: return "AI 创作工具"
        return "教育平台"

    type_colors = {"LLM 大模型": "#E64B35", "AI 创作工具": "#4DBBD5",
                   "教育平台": "#00A087"}

    # Get company for each tool
    tool_company = tool_df.groupby("工具标准名")["公司"].first().to_dict()

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = []
    for i, (name, val) in enumerate(tool_counts.items()):
        tt = tool_type(name)
        color = type_colors[tt]
        bar = ax.barh(i, val, color=color, height=0.68, alpha=0.88,
                      edgecolor="white", linewidth=0.5)
        bars.append(bar)
        share = val / total * 100
        ax.text(val + 5, i, f"{share:.1f}%", va="center",
                fontsize=7.5, fontfamily=EN_FONT, color="#555555")
        # Company name
        comp = tool_company.get(name, "")
        if comp and comp != name and comp != "未提及":
            ax.text(3, i - 0.08, f"({comp})", va="center",
                    fontsize=5.5, fontfamily=CN_FONT, color="white", alpha=0.8)

    ax.set_yticks(range(len(tool_counts)))
    ax.set_yticklabels(tool_counts.index, fontsize=8.5, fontfamily=CN_FONT)
    ax.set_xlabel("案例数量 / Case Count", fontsize=10, fontfamily=CN_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    for tt, c in type_colors.items():
        ax.barh([], [], color=c, label=tt)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Insight
    callout(ax, "豆包+DeepSeek占Top-2\n合计676例 (22.3%)",
            xy=(tool_counts.iloc[-1], len(tool_counts)-1),
            xytext=(tool_counts.iloc[-1] - 100, len(tool_counts) - 5),
            color="#3C5488")

    add_fig_number(ax, "B01", "Top-20 AI教育工具使用频次",
                   "Horizontal Bar Chart of Top 20 AI-Education Tools")
    fig.savefig(OUTDIR / "fig_b01_top20_tools.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG B02 — Technology Pathway Sankey (matplotlib)
# ═══════════════════════════════════════════════════════════════════════════
def fig_b02():
    print("  [B02] Tech Sankey …")
    # Simplified Sankey: 技术路径类型 → 应用场景(一级) → 三赋能
    sub = df[["技术路径类型", "应用场景（一级）", "三赋能分类"]].dropna()

    # Get top categories
    tech_top = sub["技术路径类型"].value_counts().head(6).index.tolist()
    scene_top = sub["应用场景（一级）"].value_counts().head(5).index.tolist()
    san_top = sub["三赋能分类"].value_counts().head(4).index.tolist()

    sub = sub[sub["技术路径类型"].isin(tech_top) &
              sub["应用场景（一级）"].isin(scene_top) &
              sub["三赋能分类"].isin(san_top)]

    # Build flow data
    # Left: tech, Middle: scene, Right: sanfuneng
    tech_list = tech_top
    scene_list = scene_top
    san_list = san_top

    # Assign vertical positions
    def positions(labels, x, total_height=8):
        counts = []
        for l in labels:
            if x == 0:
                counts.append(len(sub[sub["技术路径类型"] == l]))
            elif x == 1:
                counts.append(len(sub[sub["应用场景（一级）"] == l]))
            else:
                counts.append(len(sub[sub["三赋能分类"] == l]))
        total = sum(counts)
        gap = 0.3
        usable = total_height - gap * (len(labels) - 1)
        heights = [c / total * usable for c in counts]
        pos = []
        y = 0
        for h in heights:
            pos.append((y, h))
            y += h + gap
        return pos, heights, counts

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-0.5, 10.5)

    x_positions = [0.5, 4.5, 8.5]
    width = 0.8

    all_pos = {}  # label -> (x, y_start, height)
    node_colors = {}

    for col_idx, (labels, x) in enumerate(
            [(tech_list, 0), (scene_list, 1), (san_list, 2)]):
        pos, heights, counts = positions(labels, col_idx)
        for i, (label, (y, h), cnt) in enumerate(zip(labels, pos, counts)):
            color = PAL12[(col_idx * 4 + i) % len(PAL12)]
            node_colors[label] = color
            xp = x_positions[col_idx]
            rect = FancyBboxPatch((xp - width/2, y), width, h,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor="white",
                                  linewidth=1, alpha=0.9)
            ax.add_patch(rect)
            # Label
            short_label = label.replace("T1_", "").replace("T2_", "")\
                               .replace("T3_", "").replace("T4_", "")\
                               .replace("T5_", "").replace("T6_", "")
            if len(short_label) > 6:
                short_label = short_label[:6] + "…"
            ax.text(xp, y + h/2, f"{short_label}\n({cnt})",
                    ha="center", va="center", fontsize=7,
                    fontfamily=CN_FONT, fontweight="bold", color="white")
            all_pos[label] = (xp, y, h)

    # Draw flows (as filled polygons)
    # tech -> scene
    for tl in tech_list:
        for sl in scene_list:
            flow = len(sub[(sub["技术路径类型"] == tl) & (sub["应用场景（一级）"] == sl)])
            if flow < 5:
                continue
            tx, ty, th = all_pos[tl]
            sx, sy, sh = all_pos[sl]
            color = node_colors[tl]
            # Simple bezier band
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch
            # Normalize flow to height
            band_h = flow / 80  # scale
            band_h = min(band_h, th * 0.8, sh * 0.8)
            # Midpoints
            y1 = ty + th/2
            y2 = sy + sh/2
            verts = [
                (tx + width/2, y1 - band_h/2),
                ((tx + width/2 + sx - width/2)/2, y1 - band_h/2),
                ((tx + width/2 + sx - width/2)/2, y2 - band_h/2),
                (sx - width/2, y2 - band_h/2),
                (sx - width/2, y2 + band_h/2),
                ((tx + width/2 + sx - width/2)/2, y2 + band_h/2),
                ((tx + width/2 + sx - width/2)/2, y1 + band_h/2),
                (tx + width/2, y1 + band_h/2),
                (tx + width/2, y1 - band_h/2),
            ]
            codes = [Path.MOVETO] + [Path.CURVE4]*6 + [Path.LINETO, Path.CLOSEPOLY]
            # Simplified: just use polygon
            from matplotlib.patches import Polygon
            poly = Polygon(verts, closed=True, facecolor=color, alpha=0.15,
                           edgecolor=color, linewidth=0.3)
            ax.add_patch(poly)

    # scene -> sanfuneng
    for sl in scene_list:
        for sn in san_list:
            flow = len(sub[(sub["应用场景（一级）"] == sl) & (sub["三赋能分类"] == sn)])
            if flow < 5:
                continue
            sx, sy, sh = all_pos[sl]
            nx, ny, nh = all_pos[sn]
            color = node_colors[sn]
            band_h = flow / 80
            band_h = min(band_h, sh * 0.8, nh * 0.8)
            y1 = sy + sh/2
            y2 = ny + nh/2
            verts = [
                (sx + width/2, y1 - band_h/2),
                ((sx + width/2 + nx - width/2)/2, y1 - band_h/2),
                ((sx + width/2 + nx - width/2)/2, y2 - band_h/2),
                (nx - width/2, y2 - band_h/2),
                (nx - width/2, y2 + band_h/2),
                ((sx + width/2 + nx - width/2)/2, y2 + band_h/2),
                ((sx + width/2 + nx - width/2)/2, y1 + band_h/2),
                (sx + width/2, y1 + band_h/2),
                (sx + width/2, y1 - band_h/2),
            ]
            from matplotlib.patches import Polygon
            poly = Polygon(verts, closed=True, facecolor=color, alpha=0.15,
                           edgecolor=color, linewidth=0.3)
            ax.add_patch(poly)

    # Column headers
    for xp, label in zip(x_positions,
                          ["技术路径类型\nTech Pathway",
                           "应用场景(一级)\nScenario",
                           "三赋能分类\nEmpowerment"]):
        ax.text(xp, -0.8, label, ha="center", va="center",
                fontsize=10, fontfamily=CN_FONT, fontweight="bold", color="#333333")

    ax.set_ylim(-1.5, 9.5)
    ax.axis("off")

    # Insight
    ax.text(5, 9.2,
            "   T1内容生成 → 助学 → 赋能学生 是最主要的技术赋能路径   ",
            ha="center", fontsize=10, fontfamily=CN_FONT, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#3C5488", ec="none", alpha=0.9))

    add_fig_number(ax, "B02", "技术赋能路径桑基图",
                   "Sankey Diagram: Technology Pathway → Scenario → Empowerment")
    fig.savefig(OUTDIR / "fig_b02_tech_sankey.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG C01 — UMAP Cluster Visualization
# ═══════════════════════════════════════════════════════════════════════════
def fig_c01():
    print("  [C01] UMAP clusters …")
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder

    # Prepare features
    feat_cols = ["学段", "应用场景（一级）", "三赋能分类", "技术路径类型",
                 "智慧教育境界", "iSTAR人机协同层级"]
    num_cols = ["D1_深度学习", "D2_绿色鲁棒", "D3_循证教学", "D4_人机互信", "创新深度评分"]

    sub = df[feat_cols + num_cols].dropna().copy()
    les = {}
    for c in feat_cols:
        le = LabelEncoder()
        sub[c] = le.fit_transform(sub[c].astype(str))
        les[c] = le

    X = sub.values.astype(float)
    # Normalize
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                         min_dist=0.3, metric="euclidean")
    embedding = reducer.fit_transform(X)

    # KMeans
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Cluster names based on dominant characteristics
    cluster_names = {}
    orig_df = df[feat_cols + num_cols].dropna().copy()
    for cl in range(8):
        mask = labels == cl
        dominant_scene = orig_df.loc[mask, "应用场景（一级）"].mode()
        dominant_tech = orig_df.loc[mask, "技术路径类型"].mode()
        sc = dominant_scene.iloc[0] if len(dominant_scene) > 0 else "?"
        tc = dominant_tech.iloc[0] if len(dominant_tech) > 0 else "?"
        tc_short = tc.split("_")[-1] if "_" in tc else tc
        sc_short = sc[:4] if len(sc) > 4 else sc
        cluster_names[cl] = f"C{cl}: {sc_short}×{tc_short}"

    fig, ax = plt.subplots(figsize=(14, 10))
    for cl in range(8):
        mask = labels == cl
        color = PAL8[cl % len(PAL8)]
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=8, alpha=0.45, label=cluster_names[cl],
                   edgecolors="none")
        # Cluster region background
        cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
        from matplotlib.patches import Ellipse
        std_x = embedding[mask, 0].std() * 1.5
        std_y = embedding[mask, 1].std() * 1.5
        ell = Ellipse((cx, cy), std_x * 2, std_y * 2,
                       facecolor=color, edgecolor=color,
                       alpha=0.08, linewidth=1, linestyle="--")
        ax.add_patch(ell)
        # Cluster label
        ax.text(cx, cy, cluster_names[cl], ha="center", va="center",
                fontsize=6, fontfamily=CN_FONT, fontweight="bold",
                color=color, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.7, linewidth=0.8))

    ax.set_xlabel("UMAP-1", fontsize=10, fontfamily=EN_FONT)
    ax.set_ylabel("UMAP-2", fontsize=10, fontfamily=EN_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85, ncol=2,
              markerscale=2)

    # Insight
    ax.text(0.98, 0.04,
            "8个案例聚类揭示 AI教育应用\n的差异化发展模式",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontfamily=CN_FONT, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#E64B35", ec="none", alpha=0.9))

    add_fig_number(ax, "C01", "AI教育案例UMAP聚类分布",
                   "UMAP 2D Embedding of AI-Education Cases (K=8 Clusters)")
    fig.savefig(OUTDIR / "fig_c01_umap_clusters.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG D01 — Lorenz Curve (Market Concentration)
# ═══════════════════════════════════════════════════════════════════════════
def fig_d01():
    print("  [D01] Lorenz curve …")
    tool_counts = df["工具标准名"].value_counts()
    tool_counts = tool_counts[tool_counts.index != "未提及"]
    sorted_vals = np.sort(tool_counts.values)
    cum = np.cumsum(sorted_vals) / sorted_vals.sum()
    cum = np.insert(cum, 0, 0)
    x = np.linspace(0, 1, len(cum))

    # Gini coefficient
    n = len(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_vals) / (n * sorted_vals.sum())) - (n + 1) / n

    fig, ax = plt.subplots(figsize=(14, 10))

    # Shade inequality area
    ax.fill_between(x, x[:len(cum)], cum, alpha=0.2, color="#E64B35",
                    label="不平等区域 / Inequality Area")
    # Equality line
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.5,
            label="完全平等线 / Line of Equality")
    # Typical industry reference (Gini ~ 0.5)
    typical_y = np.power(x, 2)
    ax.plot(x, typical_y, ":", color="#4DBBD5", linewidth=1.5, alpha=0.7,
            label="典型行业参考 (Gini≈0.50)")
    # Actual Lorenz
    ax.plot(x, cum, color="#E64B35", linewidth=2.8, label="Lorenz 曲线")

    ax.set_xlabel("工具累计占比 / Cumulative Share of Tools", fontsize=11,
                  fontfamily=CN_FONT)
    ax.set_ylabel("案例累计占比 / Cumulative Share of Cases", fontsize=11,
                  fontfamily=CN_FONT)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Gini annotation
    ax.text(0.65, 0.25, f"Gini = {gini:.3f}",
            fontsize=22, fontfamily=EN_FONT, fontweight="bold",
            color="#E64B35", transform=ax.transAxes)
    ax.text(0.65, 0.18, "高度集中市场 / Highly Concentrated",
            fontsize=10, fontfamily=CN_FONT, color="#555555",
            transform=ax.transAxes)

    # CR4, CR10 markers
    total = sorted_vals.sum()
    sorted_desc = np.sort(tool_counts.values)[::-1]
    cr4 = sorted_desc[:4].sum() / total * 100
    cr10 = sorted_desc[:10].sum() / total * 100
    ax.text(0.65, 0.12, f"CR4 = {cr4:.1f}%  |  CR10 = {cr10:.1f}%",
            fontsize=10, fontfamily=EN_FONT, color="#3C5488",
            transform=ax.transAxes)

    # Insight
    ax.text(0.5, -0.10,
            '   前4款工具占据28%市场份额，AI教育工具市场呈"长尾"分布   ',
            transform=ax.transAxes, ha="center", fontsize=10,
            fontfamily=CN_FONT, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#3C5488", ec="none", alpha=0.9))

    add_fig_number(ax, "D01", "AI教育工具市场集中度洛伦兹曲线",
                   "Lorenz Curve & Gini Coefficient of AI-Education Tool Market")
    fig.savefig(OUTDIR / "fig_d01_lorenz_curve.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG E01 — 三赋能 × iSTAR Heatmap with Marginals
# ═══════════════════════════════════════════════════════════════════════════
def fig_e01():
    print("  [E01] Sanfuneng × iSTAR heatmap …")
    ct = pd.crosstab(df["三赋能分类"], df["iSTAR人机协同层级"])
    # Reorder
    san_order = ["赋能学生", "赋能教师", "赋能评价", "赋能学校"]
    istar_order = ["HUM(0)", "HMC(1)", "HM2C(2)"]
    san_order = [s for s in san_order if s in ct.index]
    istar_order = [s for s in istar_order if s in ct.columns]
    ct = ct.loc[san_order, istar_order]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main heatmap
    cmap_heat = LinearSegmentedColormap.from_list(
        "teal_red", ["#F0F9E8", "#BAE4BC", "#7BCCC4", "#43A2CA", "#0868AC"])
    im = ax_main.imshow(ct.values, cmap=cmap_heat, aspect="auto")

    # Annotate cells
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            pct = val / ct.values.sum() * 100
            color = "white" if val > ct.values.max() * 0.5 else "#333333"
            ax_main.text(j, i, f"{val}\n({pct:.1f}%)",
                         ha="center", va="center", fontsize=10,
                         fontfamily=EN_FONT, fontweight="bold", color=color)

    ax_main.set_xticks(range(len(istar_order)))
    ax_main.set_xticklabels(istar_order, fontsize=10, fontfamily=EN_FONT)
    ax_main.set_yticks(range(len(san_order)))
    ax_main.set_yticklabels(san_order, fontsize=10, fontfamily=CN_FONT)

    # Top marginal
    col_sums = ct.sum(axis=0)
    ax_top.bar(range(len(istar_order)), col_sums.values, color=PAL8[:3],
               alpha=0.85, width=0.6)
    for i, v in enumerate(col_sums.values):
        ax_top.text(i, v + 30, str(v), ha="center", fontsize=8, fontfamily=EN_FONT)
    ax_top.set_xlim(-0.5, len(istar_order) - 0.5)
    ax_top.axis("off")

    # Right marginal
    row_sums = ct.sum(axis=1)
    ax_right.barh(range(len(san_order)), row_sums.values, color=PAL8[:4],
                  alpha=0.85, height=0.6)
    for i, v in enumerate(row_sums.values):
        ax_right.text(v + 30, i, str(v), va="center", fontsize=8, fontfamily=EN_FONT)
    ax_right.set_ylim(-0.5, len(san_order) - 0.5)
    ax_right.invert_yaxis()
    ax_right.axis("off")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_right, fraction=0.3, pad=0.2, shrink=0.8)
    cbar.set_label("案例数", fontsize=8, fontfamily=CN_FONT)

    # Insight
    # Compute HUM(0) + HMC(1) percentage
    low_levels = ct[["HUM(0)", "HMC(1)"]].sum().sum() if "HUM(0)" in ct.columns else ct[["HMC(1)"]].sum().sum()
    low_pct = low_levels / ct.values.sum() * 100
    fig.text(0.5, 0.02,
             f"   Level 0-1 占比 {low_pct:.0f}%：人机协同尚处于初级阶段   ",
             ha="center", fontsize=11, fontfamily=CN_FONT, fontweight="bold",
             color="white",
             bbox=dict(boxstyle="round,pad=0.5", fc="#E64B35", ec="none", alpha=0.9))

    add_fig_number_fig(fig, "E01", "三赋能 × iSTAR人机协同层级热力图",
                       "Heatmap: Empowerment × iSTAR Human-AI Collaboration Level")
    fig.savefig(OUTDIR / "fig_e01_sanfuneng_istar.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG E02 — Ridgeline / Joy Plot of Innovation Score by Province
# ═══════════════════════════════════════════════════════════════════════════
def fig_e02():
    print("  [E02] Innovation ridgeline …")
    top_provinces = df["省份"].value_counts().head(10).index.tolist()
    sub = df[df["省份"].isin(top_provinces)].copy()

    fig, axes = plt.subplots(10, 1, figsize=(14, 12), sharex=True)
    fig.subplots_adjust(hspace=-0.3)

    x_range = np.linspace(0.5, 5.5, 200)

    for i, prov in enumerate(top_provinces):
        ax = axes[i]
        vals = sub[sub["省份"] == prov]["创新深度评分"].dropna().values
        if len(vals) < 5:
            ax.axis("off")
            continue

        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(vals, bw_method=0.4)
            density = kde(x_range)
        except Exception:
            density = np.zeros_like(x_range)

        color = PAL12[i % len(PAL12)]
        ax.fill_between(x_range, density, alpha=0.6, color=color)
        ax.plot(x_range, density, color=color, linewidth=1.5)

        # Mean line
        mean_val = vals.mean()
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(mean_val + 0.05, density.max() * 0.7,
                f"μ={mean_val:.2f}", fontsize=7, fontfamily=EN_FONT,
                color=color, fontweight="bold")

        ax.text(-0.01, 0.5, prov.replace("省", "").replace("市", "")
                                  .replace("自治区", "").replace("维吾尔", ""),
                transform=ax.transAxes, ha="right", va="center",
                fontsize=9, fontfamily=CN_FONT, fontweight="bold", color=color)

        ax.set_yticks([])
        ax.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("创新深度评分 / Innovation Depth Score", fontsize=10,
                        fontfamily=CN_FONT)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].spines["bottom"].set_color("#CCCCCC")

    # Insight
    fig.text(0.5, 0.01,
             "   各省创新深度评分整体集中在2-3分区间，省际差异显著   ",
             ha="center", fontsize=10, fontfamily=CN_FONT, fontweight="bold",
             color="white",
             bbox=dict(boxstyle="round,pad=0.5", fc="#00A087", ec="none", alpha=0.9))

    add_fig_number_fig(fig, "E02", "Top-10省份创新深度评分分布脊线图",
                       "Ridgeline Plot of Innovation Depth Score by Top 10 Provinces")
    fig.savefig(OUTDIR / "fig_e02_innovation_ridgeline.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG F01 — Random Forest Feature Importance (SHAP)
# ═══════════════════════════════════════════════════════════════════════════
def fig_f01():
    print("  [F01] RF SHAP importance …")
    shap_data = causal["10_predictive_modeling"]["shap_analysis"]["mean_abs_shap"]
    fi_data = causal["10_predictive_modeling"]["feature_importance_top15"]

    # Use SHAP values
    features = list(shap_data.keys())
    values = list(shap_data.values())
    # Sort ascending for horizontal bar
    idx = np.argsort(values)
    features = [features[i] for i in idx]
    values = [values[i] for i in idx]

    # Nice labels
    label_map = {
        "自研_num": "自主研发 (Self-dev)",
        "D3_循证教学": "循证教学 (Evidence-based)",
        "iSTAR_num": "iSTAR 数值",
        "iSTAR人机协同层级": "iSTAR 协同层级",
        "D1_深度学习": "深度学习 (Deep Learning)",
        "D4_人机互信": "人机互信 (Human-AI Trust)",
        "产品分类": "产品分类 (Product Type)",
        "应用场景（一级）": "应用场景 (Scenario)",
        "三赋能分类": "三赋能 (Empowerment)",
        "产品技术代际": "产品技术代际 (Tech Gen.)",
        "学段_clean": "学段 (School Stage)",
        "tech_gen_num": "技术代际编码",
        "区域": "区域 (Region)",
        "产品形态": "产品形态 (Product Form)",
        "dev_level": "发展水平 (Dev Level)",
    }
    nice_features = [label_map.get(f, f) for f in features]

    cmap_bar = LinearSegmentedColormap.from_list(
        "shap_grad", ["#4DBBD5", "#00A087", "#F39B7F", "#E64B35"])
    norm = Normalize(0, len(features) - 1)
    colors = [cmap_bar(norm(i)) for i in range(len(features))]

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(range(len(features)), values, color=colors, height=0.65,
                   edgecolor="white", linewidth=0.5, alpha=0.9)

    # Value labels
    for i, (v, f) in enumerate(zip(values, features)):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center",
                fontsize=7.5, fontfamily=EN_FONT, color="#555555")

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(nice_features, fontsize=9, fontfamily=CN_FONT)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11, fontfamily=EN_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Insight
    callout(ax, "自主研发是最强预测变量\n重要性远超其他特征",
            xy=(values[-1], len(features) - 1),
            xytext=(values[-1] * 0.5, len(features) - 4),
            color="#E64B35")

    add_fig_number(ax, "F01", "随机森林SHAP特征重要性",
                   "Random Forest SHAP Feature Importance for Innovation Prediction")
    fig.savefig(OUTDIR / "fig_f01_rf_shap.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# FIG E03 — Three Realms Pyramid Infographic
# ═══════════════════════════════════════════════════════════════════════════
def fig_e03():
    print("  [E03] Three realms pyramid …")
    realm_counts = df["智慧教育境界"].value_counts()
    total = realm_counts.sum()

    # Expected order: 第一 (bottom), 第二 (middle), 第三 (top)
    realms = [
        ("第一境界_智慧环境", "第一境界\n智慧环境", "Smart\nEnvironment",
         "#4DBBD5", realm_counts.get("第一境界_智慧环境", 0)),
        ("第二境界_教学模式", "第二境界\n教学模式", "Teaching\nModel",
         "#00A087", realm_counts.get("第二境界_教学模式", 0)),
        ("第三境界_制度变革", "第三境界\n制度变革", "Institutional\nReform",
         "#E64B35", realm_counts.get("第三境界_制度变革", 0)),
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw pyramid layers (bottom to top)
    layer_heights = [2.8, 2.8, 2.8]
    layer_widths = [10, 7, 4]  # bottom wider
    y_base = 0.5

    for i, (key, cn_label, en_label, color, cnt) in enumerate(realms):
        y = y_base + sum(layer_heights[:i])
        w = layer_widths[i]
        h = layer_heights[i]
        pct = cnt / total * 100

        # Trapezoid (wider at bottom)
        if i < 2:
            w_next = layer_widths[i + 1]
        else:
            w_next = 1.0

        # Draw as trapezoid
        verts = [
            (-w/2, y),
            (w/2, y),
            (w_next/2, y + h),
            (-w_next/2, y + h),
        ]
        from matplotlib.patches import Polygon
        trap = Polygon(verts, closed=True, facecolor=color, edgecolor="white",
                       linewidth=2, alpha=0.88)
        ax.add_patch(trap)

        # Gradient overlay
        trap_light = Polygon(verts, closed=True,
                             facecolor=lighten(color, 0.3),
                             edgecolor="none", alpha=0.3)
        ax.add_patch(trap_light)

        # Labels
        ax.text(0, y + h/2, cn_label, ha="center", va="center",
                fontsize=14, fontfamily=CN_FONT, fontweight="bold",
                color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground=color)])
        ax.text(w/2 + 0.8, y + h/2, en_label, ha="left", va="center",
                fontsize=9, fontfamily=EN_FONT, color=color, style="italic")

        # Count + percentage
        ax.text(-w/2 - 0.5, y + h/2,
                f"{cnt:,}例\n({pct:.1f}%)",
                ha="right", va="center", fontsize=11, fontfamily=CN_FONT,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, linewidth=1.5, alpha=0.9))

    # Top arrow / aspirational icon
    ax.annotate("", xy=(0, y_base + sum(layer_heights) + 0.8),
                xytext=(0, y_base + sum(layer_heights)),
                arrowprops=dict(arrowstyle="-|>", color="#E64B35",
                                lw=2.5))
    ax.text(0, y_base + sum(layer_heights) + 1.2, "智慧教育愿景",
            ha="center", fontsize=12, fontfamily=CN_FONT, fontweight="bold",
            color="#E64B35")

    # Insight box
    ax.text(0, -0.5,
            "   第二境界占比最高(59.8%)，第三境界仅6.6%——制度变革仍是瓶颈   ",
            ha="center", fontsize=10, fontfamily=CN_FONT, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", fc="#3C5488", ec="none", alpha=0.9))

    add_fig_number(ax, "E03", "智慧教育三重境界分布金字塔",
                   "Three Realms of Smart Education – Distribution Pyramid")
    fig.savefig(OUTDIR / "fig_e03_three_realms.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(" AI-Education Research · Part 1 Figures")
    print("=" * 60)

    funcs = [
        ("A01", fig_a01),
        ("A02", fig_a02),
        ("A03", fig_a03),
        ("A04", fig_a04),
        ("B01", fig_b01),
        ("B02", fig_b02),
        ("C01", fig_c01),
        ("D01", fig_d01),
        ("E01", fig_e01),
        ("E02", fig_e02),
        ("F01", fig_f01),
        ("E03", fig_e03),
    ]

    success = 0
    for name, func in funcs:
        try:
            func()
            print(f"  ✓ {name} done")
            success += 1
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f" Completed: {success}/{len(funcs)} figures")
    print(f" Output: {OUTDIR}")
    print("=" * 60)
