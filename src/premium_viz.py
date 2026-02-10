#!/usr/bin/env python3
"""Premium visualization redesign for AI Education Research.
Generates Nature/Science quality figures with Chinese captions.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
import squarify
from wordcloud import WordCloud
import networkx as nx
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
CSV_PATH = '/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv'
OUTPUT_DIR = Path('/Users/sakai/Desktop/产业调研/ai-edu-research/output')
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'font.size': 11,
})

# ── Color Palette ──────────────────────────────────────────────────────
PRIMARY = '#1B4F72'
SECONDARY = '#2E86C1'
ACCENT = '#E74C3C'
WARM = '#F39C12'
TEAL = '#17A589'
PURPLE = '#8E44AD'
GRADIENT_BLUES = ['#D6EAF8', '#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1', '#2874A6', '#21618C', '#1B4F72', '#154360']
PALETTE_6 = [PRIMARY, SECONDARY, ACCENT, WARM, TEAL, PURPLE]
PALETTE_8 = [PRIMARY, SECONDARY, TEAL, '#27AE60', WARM, '#E67E22', ACCENT, PURPLE]

def _load_json(name):
    fp = OUTPUT_DIR / name
    return json.loads(fp.read_bytes().decode('utf-8-sig'))

def _remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _add_caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha='center', va='top', fontsize=10,
             fontstyle='italic', color='#555555',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                       edgecolor='#DEE2E6', alpha=0.9))

def _add_analysis(fig, text, x=0.02, y=0.02):
    """Add an analysis insight annotation box."""
    fig.text(x, y, text, ha='left', va='bottom', fontsize=8,
             color='#1a5276', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                       edgecolor='#cccccc', alpha=0.8))

def _save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  [OK] {name}')

# ── Load Data ──────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
province_data = _load_json('province_distribution.json')
stage_data = _load_json('stage_distribution.json')
tool_data = _load_json('tool_product_distribution.json')
subject_data = _load_json('subject_distribution.json')
scenario_data = _load_json('scenario_analysis.json')
cooccurrence_data = _load_json('nlp_cooccurrence.json')
lda_data = _load_json('nlp_lda_topics.json')
pathway_data = _load_json('nlp_tech_pathways.json')
maturity_data = _load_json('industry_maturity.json')
selfdev_data = _load_json('self_developed_ratio.json')
market_data = _load_json('company_market_share.json')
cross_data = _load_json('cross_tabulations.json')
tfidf_data = _load_json('nlp_tfidf_keywords.json')
tech_elements = _load_json('nlp_tech_elements.json')
cultivation_data = _load_json('nlp_cultivation.json')

# ═══════════════════════════════════════════════════════════════════════
# FIG 01 — Province Distribution (Gradient Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════
def fig01():
    print('Fig01: Province heatmap...')
    prov = province_data['case_count_by_province']
    prov = {k: v for k, v in prov.items() if k != '未提及'}
    names = list(prov.keys())
    vals = list(prov.values())
    total = sum(vals)
    # Sort ascending for horizontal bar
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 10))
    norm = plt.Normalize(min(vals), max(vals))
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_grad', ['#D6EAF8', PRIMARY])
    colors = [cmap(norm(v)) for v in vals]

    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('案例数量', fontsize=11, fontweight='bold')
    ax.set_title('AI教育案例省域分布', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    _remove_spines(ax)

    # Percentage labels
    for i, (bar, v) in enumerate(zip(bars, vals)):
        pct = v / total * 100
        ax.text(v + total * 0.005, i, f'{v}  ({pct:.1f}%)', va='center', fontsize=7.5, color='#333')

    # Top 5 callout box
    top5 = list(reversed(names[-5:]))
    top5_vals = list(reversed(vals[-5:]))
    top5_text = 'Top 5 省份\n' + '\n'.join(
        [f'  {i+1}. {n}: {v}例 ({v/total*100:.1f}%)' for i, (n, v) in enumerate(zip(top5, top5_vals))])
    props = dict(boxstyle='round,pad=0.6', facecolor='#FEF9E7', edgecolor=WARM, alpha=0.95)
    ax.text(0.97, 0.05, top5_text, transform=ax.transAxes, fontsize=8.5,
            va='bottom', ha='right', bbox=props, color='#333', linespacing=1.5)

    # Regional annotation
    regions = {'华东': ['浙江省','上海市','江苏省','福建省','山东省','安徽省','江西省'],
               '华北': ['北京市','天津市','河北省','山西省','内蒙古自治区'],
               '西南': ['四川省','重庆市','贵州省'],
               '西北': ['宁夏回族自治区','新疆维吾尔自治区','陕西省','甘肃省'],
               '东北': ['吉林省','黑龙江省','辽宁省']}
    region_counts = {}
    for rname, provs in regions.items():
        region_counts[rname] = sum(prov.get(p, 0) for p in provs)
    region_text = '区域汇总: ' + ' | '.join([f'{k} {v}例' for k, v in
                  sorted(region_counts.items(), key=lambda x: -x[1])])
    ax.text(0.5, 1.02, region_text, transform=ax.transAxes, ha='center',
            fontsize=8, color='#666', fontstyle='italic')

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图1 AI教育案例省域分布（N=1,690）', y=0.01)
    _add_analysis(fig, '省域分布呈现显著的东部集聚效应，浙江、北京等教育信息化先发地区案例数量领先。', x=0.12, y=0.08)
    _save(fig, 'fig01_province_heatmap.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 02 — School Stage Donut
# ═══════════════════════════════════════════════════════════════════════
def fig02():
    print('Fig02: School stage donut...')
    raw = stage_data['case_count_by_stage']
    # Keep main stages
    main_stages = ['小学', '初中', '高中', '幼儿园', '中学', '其他']
    other_count = 0
    stages, counts = [], []
    for k, v in raw.items():
        if k in ['小学', '初中', '高中', '幼儿园']:
            stages.append(k)
            counts.append(v)
        else:
            other_count += v
    if other_count > 0:
        stages.append('其他')
        counts.append(other_count)
    total = sum(counts)

    colors_donut = [PRIMARY, SECONDARY, TEAL, WARM, '#BDC3C7']
    explode = [0.03] * len(stages)

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts = ax.pie(counts, labels=None, colors=colors_donut[:len(stages)],
                           startangle=90, explode=explode,
                           wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))

    # Inner text
    ax.text(0, 0.05, f'N = {total}', ha='center', va='center',
            fontsize=28, fontweight='bold', color=PRIMARY)
    avg_tools = stage_data['tool_count_by_stage']
    total_tools = sum(avg_tools.values())
    ax.text(0, -0.08, f'涉及工具 {total_tools} 次', ha='center', va='center',
            fontsize=12, color='#666')

    # Leader lines with labels
    for i, (wedge, stage, count) in enumerate(zip(wedges, stages, counts)):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(ang))
        y = np.sin(np.radians(ang))
        pct = count / total * 100
        ha = 'left' if x > 0 else 'right'
        ax.annotate(f'{stage}\n{count}例 ({pct:.1f}%)',
                    xy=(0.8 * x, 0.8 * y),
                    xytext=(1.3 * x, 1.3 * y),
                    fontsize=11, fontweight='bold', color=colors_donut[i],
                    ha=ha, va='center',
                    arrowprops=dict(arrowstyle='-', color='#999', lw=0.8))

    ax.set_title('学段分布结构', fontsize=16, fontweight='bold', color=PRIMARY, pad=30)
    fig.subplots_adjust(bottom=0.08)
    _add_caption(fig, '图2 学段分布结构（N=1,690）', y=0.01)
    _add_analysis(fig, '小学阶段AI教育应用最为活跃，占比最高，反映基础教育数字化转型的重心。', x=0.05, y=0.08)
    _save(fig, 'fig02_school_stage_donut.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 03 — Top 20 Tools Lollipop Chart
# ═══════════════════════════════════════════════════════════════════════
def fig03():
    print('Fig03: Top 20 tools lollipop...')
    top30 = tool_data['top30_tools']
    items = list(top30.items())[:20]
    names = [x[0] for x in items][::-1]
    vals = [x[1] for x in items][::-1]
    total = sum(top30.values())

    # Categorize tools
    llm_tools = {'豆包', 'DeepSeek 大模型', '文心一言', 'Kimi', '通义千问大模型',
                 '星火大模型', '智谱清言', '腾讯元宝', '讯飞星火'}
    edu_tools = {'希沃白板', '国家智慧教育平台', '文曲智阅', '央馆领航AI',
                 '希沃AI', '班级优化大师', '智学网', '希沃易课堂', '之江汇AI'}
    cat_colors = []
    for n in names:
        if n in llm_tools:
            cat_colors.append(ACCENT)
        elif n in edu_tools:
            cat_colors.append(TEAL)
        else:
            cat_colors.append(SECONDARY)

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = range(len(names))

    # Lines
    for i, v in enumerate(vals):
        ax.plot([0, v], [i, i], color=cat_colors[i], linewidth=1.5, alpha=0.6)

    # Dots - top 3 highlighted
    for i, (v, c) in enumerate(zip(vals, cat_colors)):
        rank = len(names) - i
        size = 120 if rank <= 3 else 60
        ec = WARM if rank <= 3 else 'white'
        lw = 2 if rank <= 3 else 1
        ax.scatter(v, i, s=size, color=c, edgecolors=ec, linewidths=lw, zorder=5)

    # Labels
    for i, v in enumerate(vals):
        pct = v / total * 100
        ax.text(v + total * 0.008, i, f'{v} ({pct:.1f}%)', va='center', fontsize=8, color='#333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('提及次数', fontsize=11, fontweight='bold')
    ax.set_title('AI教育工具Top20及市场份额', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    _remove_spines(ax)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT, markersize=8, label='大模型'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TEAL, markersize=8, label='教育平台'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SECONDARY, markersize=8, label='通用工具'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=WARM, markersize=10,
               markeredgecolor=WARM, markeredgewidth=2, label='Top 3'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图3 AI教育工具Top20及市场份额', y=0.01)
    _add_analysis(fig, '大模型类工具（豆包、DeepSeek等）快速崛起，与传统教育平台形成双轨并行格局。', x=0.12, y=0.08)
    _save(fig, 'fig03_top20_tools_bar.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 04 — Product Ecosystem Treemap
# ═══════════════════════════════════════════════════════════════════════
def fig04():
    print('Fig04: Product ecosystem treemap...')
    pf = tool_data['product_form']
    # Group small categories
    items = sorted(pf.items(), key=lambda x: -x[1])
    main_items = [(k, v) for k, v in items if v >= 10]
    other_v = sum(v for k, v in items if v < 10)
    if other_v > 0:
        main_items.append(('其他', other_v))

    labels = [x[0] for x in main_items]
    sizes = [x[1] for x in main_items]
    total = sum(sizes)

    cmap = plt.cm.get_cmap('Blues_r')
    norm = plt.Normalize(0, len(labels))
    colors = [cmap(norm(i)) for i in range(len(labels))]
    # Mix in some warm colors for variety
    accent_indices = [2, 5, 8, 11]
    accent_cmaps = [ACCENT, WARM, TEAL, PURPLE]
    for idx, ac in zip(accent_indices, accent_cmaps):
        if idx < len(colors):
            colors[idx] = ac

    fig, ax = plt.subplots(figsize=(14, 9))
    rects = squarify.plot(sizes=sizes, label=None, color=colors, alpha=0.85,
                          ax=ax, bar_kwargs=dict(linewidth=2, edgecolor='white'))

    # Add labels manually for better control
    normed = squarify.normalize_sizes(sizes, 100, 100)
    rects_coords = squarify.squarify(normed, 0, 0, 100, 100)
    for rc, label, size in zip(rects_coords, labels, sizes):
        x_c = rc['x'] + rc['dx'] / 2
        y_c = rc['y'] + rc['dy'] / 2
        pct = size / total * 100
        if rc['dx'] > 8 and rc['dy'] > 6:
            ax.text(x_c, y_c, f'{label}\n{size} ({pct:.1f}%)',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if size > 100 else '#333',
                    transform=ax.transData)
        elif rc['dx'] > 4 and rc['dy'] > 3:
            ax.text(x_c, y_c, f'{label}\n{size}',
                    ha='center', va='center', fontsize=6, color='white',
                    transform=ax.transData)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_title('教育科技产品生态图谱', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图4 教育科技产品生态图谱', y=0.01)
    _add_analysis(fig, '产品形态以软件平台为主导，硬件类产品占比较低，反映AI教育以轻量化部署为主。', x=0.05, y=0.08)
    _save(fig, 'fig04_product_ecosystem_treemap.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 05 — Scenario Sunburst (Nested Donut)
# ═══════════════════════════════════════════════════════════════════════
def fig05():
    print('Fig05: Scenario sunburst...')
    l1 = scenario_data['scenario_level1']
    l2 = scenario_data['scenario_level2']
    # Remove 未提及
    l1 = {k: v for k, v in l1.items() if k != '未提及'}
    l2 = {k: v for k, v in l2.items() if k != '未提及'}

    # Inner ring: L1
    l1_names = list(l1.keys())
    l1_vals = list(l1.values())
    l1_total = sum(l1_vals)

    # Outer ring: L2 (top 12 + other)
    l2_sorted = sorted(l2.items(), key=lambda x: -x[1])
    l2_main = l2_sorted[:12]
    l2_other = sum(v for _, v in l2_sorted[12:])
    if l2_other > 0:
        l2_main.append(('其他', l2_other))
    l2_names = [x[0] for x in l2_main]
    l2_vals = [x[1] for x in l2_main]

    # Color schemes
    warm_cool = plt.cm.get_cmap('RdYlBu_r')
    inner_colors = [warm_cool(i / max(1, len(l1_names) - 1)) for i in range(len(l1_names))]
    outer_colors = [warm_cool(i / max(1, len(l2_names) - 1)) for i in range(len(l2_names))]

    fig, ax = plt.subplots(figsize=(12, 12))

    # Inner donut
    wedges1, _ = ax.pie(l1_vals, radius=0.6, colors=inner_colors,
                        wedgeprops=dict(width=0.25, edgecolor='white', linewidth=1.5),
                        startangle=90)
    # Outer donut
    wedges2, _ = ax.pie(l2_vals, radius=0.95, colors=outer_colors,
                        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),
                        startangle=90)

    # Inner labels
    for w, name, val in zip(wedges1, l1_names, l1_vals):
        ang = (w.theta1 + w.theta2) / 2
        x = 0.47 * np.cos(np.radians(ang))
        y = 0.47 * np.sin(np.radians(ang))
        pct = val / l1_total * 100
        ax.text(x, y, f'{name}\n{pct:.0f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='#333')

    # Outer labels for major segments
    l2_total = sum(l2_vals)
    for w, name, val in zip(wedges2, l2_names, l2_vals):
        pct = val / l2_total * 100
        if pct >= 3:
            ang = (w.theta1 + w.theta2) / 2
            x = 1.15 * np.cos(np.radians(ang))
            y = 1.15 * np.sin(np.radians(ang))
            ha = 'left' if x > 0 else 'right'
            ax.text(x, y, f'{name} ({pct:.1f}%)', ha=ha, va='center',
                    fontsize=7.5, color='#444')

    ax.set_title('应用场景层级结构（L1→L2）', fontsize=16, fontweight='bold',
                 color=PRIMARY, pad=30)
    fig.subplots_adjust(bottom=0.08)
    _add_caption(fig, '图5 应用场景层级结构（L1→L2）', y=0.01)
    _add_analysis(fig, '"助学"和"助教"场景占据主导地位，AI技术在教学核心环节的渗透率最高。', x=0.05, y=0.08)
    _save(fig, 'fig05_scenario_sunburst.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 06 — Subject Distribution (Grouped Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════
def fig06():
    print('Fig06: Subject distribution...')
    raw = subject_data['case_count_by_subject']
    # Filter to single-subject entries with count >= 4
    subj = {k: v for k, v in raw.items() if k != '未提及' and '/' not in k
            and '、' not in k and ',' not in k and v >= 4}
    # Categorize
    main_subj = {'语文', '数学', '英语'}
    secondary = {'物理', '化学', '生物', '历史', '地理', '政治', '道德与法治', '思政'}
    arts_pe = {'美术', '音乐', '体育', '书法', '艺术', '美育'}
    comprehensive = {'科学', '信息科技', '综合', '信息技术', '综合实践', '人工智能',
                     '劳动教育', '心理健康教育', '德育', '安全教育', '综合素质',
                     '健康教育', '劳动', '通用技术', '科技教育', '特殊教育'}

    def get_cat(s):
        if s in main_subj: return '主科'
        if s in secondary: return '理科/文科'
        if s in arts_pe: return '艺体'
        return '综合/其他'

    cat_order = ['主科', '理科/文科', '艺体', '综合/其他']
    cat_colors = {'主科': PRIMARY, '理科/文科': SECONDARY, '艺体': WARM, '综合/其他': TEAL}

    # Sort by count within category
    items = sorted(subj.items(), key=lambda x: -x[1])
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    cats = [get_cat(n) for n in names]
    colors = [cat_colors[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = range(len(names))
    names_r = names[::-1]
    vals_r = vals[::-1]
    colors_r = colors[::-1]

    bars = ax.barh(y_pos, vals_r, color=colors_r, edgecolor='white', linewidth=0.5, height=0.7)

    # Category dots
    for i, (v, c) in enumerate(zip(vals_r, colors_r)):
        ax.scatter(-8, i, color=c, s=40, zorder=5, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_r, fontsize=9)
    ax.set_xlabel('案例数量', fontsize=11, fontweight='bold')
    ax.set_title('学科渗透格局', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    _remove_spines(ax)

    # Value labels
    for i, v in enumerate(vals_r):
        ax.text(v + 2, i, str(v), va='center', fontsize=8, color='#333')

    # Legend
    legend_elements = [mpatches.Patch(facecolor=cat_colors[c], label=c) for c in cat_order]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图6 学科渗透格局（按案例数排序）', y=0.01)
    _add_analysis(fig, '语文、数学等主科AI渗透率最高，艺体类学科正加速追赶，学科覆盖面持续扩大。', x=0.12, y=0.08)
    _save(fig, 'fig06_subject_distribution.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 07 — 五育 Radar (Multi-stage overlay)
# ═══════════════════════════════════════════════════════════════════════
def fig07():
    print('Fig07: Cultivation radar...')
    cult = scenario_data['cultivation_direction']
    # Main 五育
    wuyu = ['智育', '德育', '美育', '体育', '劳育']
    # Map labels
    cult_mapped = {}
    for k, v in cult.items():
        for w in wuyu:
            if w in k:
                cult_mapped[w] = cult_mapped.get(w, 0) + v

    # Per-stage cultivation from CSV
    stages_want = ['小学', '初中', '高中', '幼儿园']
    stage_colors = {
        '小学': PRIMARY, '初中': SECONDARY, '高中': ACCENT, '幼儿园': WARM
    }

    stage_cult = {s: {w: 0 for w in wuyu} for s in stages_want}
    for _, row in df.iterrows():
        st = str(row.get('学段', ''))
        cd = str(row.get('培养方向', ''))
        if st in stages_want:
            for w in wuyu:
                if w in cd:
                    stage_cult[st][w] += 1

    # Normalize each stage to 0-1
    for st in stages_want:
        mx = max(stage_cult[st].values()) if max(stage_cult[st].values()) > 0 else 1
        for w in wuyu:
            stage_cult[st][w] = stage_cult[st][w] / mx

    angles = np.linspace(0, 2 * np.pi, len(wuyu), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for st in stages_want:
        values = [stage_cult[st][w] for w in wuyu]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, alpha=0.8,
                color=stage_colors[st], label=st, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=stage_colors[st])

    # Ideal balance line
    ideal = [0.8] * len(wuyu) + [0.8]
    ax.plot(angles, ideal, '--', color='#999', linewidth=1.5, alpha=0.6, label='理想均衡线')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(wuyu, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color='#999')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
    ax.set_title('各学段五育融合雷达图', fontsize=16, fontweight='bold', color=PRIMARY, pad=30)
    ax.grid(True, alpha=0.2)

    fig.subplots_adjust(bottom=0.08)
    _add_caption(fig, '图7 各学段五育融合雷达图', y=0.01)
    _add_analysis(fig, '各学段五育发展不均衡，智育占绝对主导，德育和美育有待加强，劳育渗透率最低。', x=0.05, y=0.08)
    _save(fig, 'fig07_cultivation_radar.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 08 — Word Cloud (Circular mask, blue-teal colormap)
# ═══════════════════════════════════════════════════════════════════════
def fig08():
    print('Fig08: Tech word cloud...')
    keywords = tfidf_data['innovation_case']['top_keywords']
    word_freq = {kw['word']: kw.get('score', kw.get('tfidf', 1)) for kw in keywords[:150]}

    # Also add tech elements
    for elem in tech_elements['case_level_elements'][:50]:
        w = elem['element']
        if w not in word_freq:
            word_freq[w] = elem['count']

    # Circular mask
    x, y = np.ogrid[-300:300, -300:300]
    mask = (x**2 + y**2 > 290**2).astype(np.uint8) * 255

    def blue_teal_color(word, font_size, position, orientation, random_state=None, **kwargs):
        colors_list = ['#1B4F72', '#2E86C1', '#17A589', '#148F77', '#1A5276', '#2874A6', '#21618C']
        return colors_list[hash(word) % len(colors_list)]

    wc = WordCloud(width=800, height=800, background_color='white',
                   mask=mask, max_words=120, max_font_size=80,
                   min_font_size=8, color_func=blue_teal_color,
                   font_path=None, prefer_horizontal=0.7,
                   relative_scaling=0.5)
    wc.generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    # Border circle
    circle = plt.Circle((300, 300), 292, fill=False, edgecolor=PRIMARY,
                         linewidth=3, transform=ax.transData)
    ax.add_patch(circle)

    ax.set_title('AI教育创新关键词云图', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    fig.subplots_adjust(bottom=0.08)
    _add_caption(fig, '图8 AI教育创新关键词云图', y=0.01)
    _add_analysis(fig, '关键词云揭示"个性化""智能""数据驱动"等为AI教育创新的核心主题词。', x=0.05, y=0.08)
    _save(fig, 'fig08_tech_wordcloud.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 09 — Co-occurrence Network Graph
# ═══════════════════════════════════════════════════════════════════════
def fig09():
    print('Fig09: Co-occurrence network...')
    net = cooccurrence_data['network']
    nodes = net['nodes']
    edges = net['edges']

    G = nx.Graph()
    for n in nodes:
        G.add_node(n['id'], freq=n['freq'])
    for e in edges:
        G.add_edge(e['source'], e['target'], weight=e['weight'])

    # Community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        node_community = {}
        for i, comm in enumerate(communities):
            for n in comm:
                node_community[n] = i
    except Exception:
        node_community = {n: 0 for n in G.nodes()}

    comm_colors = [PRIMARY, ACCENT, TEAL, WARM, PURPLE, SECONDARY, '#27AE60', '#E67E22']
    node_colors = [comm_colors[node_community.get(n, 0) % len(comm_colors)] for n in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Degree centrality for node size
    deg = nx.degree_centrality(G)
    node_sizes = [300 + 2000 * deg[n] for n in G.nodes()]

    # Edge widths
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 3 * w / max_w for w in weights]

    fig, ax = plt.subplots(figsize=(14, 12))

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.2, edge_color='#999')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                           alpha=0.85, edgecolors='white', linewidths=0.5)

    # Labels for top 15 nodes by degree
    top_nodes = sorted(G.nodes(), key=lambda n: deg[n], reverse=True)[:15]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold',
                            font_color='#222')

    ax.axis('off')
    ax.set_title('技术要素共现网络图', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    # Legend for node size
    for s, label in [(200, '低'), (600, '中'), (1200, '高')]:
        ax.scatter([], [], s=s, c='#999', alpha=0.6, label=f'中心度: {label}')
    # Community legend
    n_comm = len(set(node_community.values()))
    for i in range(min(n_comm, 4)):
        ax.scatter([], [], s=80, c=comm_colors[i], label=f'社区 {i+1}')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9, ncol=2)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图9 技术要素共现网络图', y=0.01)
    _add_analysis(fig, '共现网络揭示技术要素间的协同关系，核心节点连接度高，形成多个技术社区。', x=0.05, y=0.08)
    _save(fig, 'fig09_tech_cooccurrence_network.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 10 — Stage × Subject Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig10():
    print('Fig10: Stage × Subject heatmap...')
    ct = cross_data['stage_subject']
    idx = ct['index']
    cols = ct['columns']
    data = np.array(ct['data'])

    # Filter to main stages and top subjects
    main_stages = ['小学', '初中', '高中', '幼儿园']
    top_subjects = cols[:12]  # top 12 subjects

    stage_idx = [idx.index(s) for s in main_stages if s in idx]
    subj_idx = list(range(min(12, len(cols))))

    filtered = data[np.ix_(stage_idx, subj_idx)]
    row_labels = [idx[i] for i in stage_idx]
    col_labels = [cols[i] for i in subj_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_orange',
        ['#FDFEFE', '#D6EAF8', '#85C1E9', '#2E86C1', '#1B4F72', '#E67E22', '#E74C3C'])
    im = ax.imshow(filtered, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    # Value annotations
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            v = filtered[i, j]
            if v > 0:
                color = 'white' if v > filtered.max() * 0.5 else '#333'
                ax.text(j, i, str(int(v)), ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('案例数', fontsize=10)

    ax.set_title('学段×学科交叉分布热力图', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    ax.spines[:].set_visible(False)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图10 学段×学科交叉分布热力图', y=0.01)
    _add_analysis(fig, '热力图显示小学语文、小学数学为AI应用最密集的学段-学科组合。', x=0.05, y=0.08)
    _save(fig, 'fig10_stage_subject_heatmap.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 11 — Industry Maturity (Horizontal Stacked Pipeline)
# ═══════════════════════════════════════════════════════════════════════
def fig11():
    print('Fig11: Industry maturity...')
    raw = maturity_data['maturity_keyword_freq']
    # Group into maturity stages
    stages_map = {
        '探索期': ['探索', '起步'],
        '初期应用': ['初期'],
        '发展期': ['发展', '成长'],
        '规模化': ['规模化', '常态化'],
        '深度融合': ['深度融合'],
        '成熟期': ['成熟'],
    }
    stage_vals = {}
    for stage, keys in stages_map.items():
        stage_vals[stage] = sum(raw.get(k, 0) for k in keys)

    stage_names = list(stages_map.keys())
    vals = [stage_vals[s] for s in stage_names]
    total = sum(vals)

    # Gradient from light to dark
    grad_colors = ['#D5F5E3', '#82E0AA', '#2ECC71', '#27AE60', '#1E8449', '#145A32']

    fig, ax = plt.subplots(figsize=(14, 5))

    # Stacked horizontal bar
    left = 0
    for i, (name, v) in enumerate(zip(stage_names, vals)):
        pct = v / total * 100
        bar = ax.barh(0, v, left=left, color=grad_colors[i], edgecolor='white',
                       linewidth=1, height=0.6)
        # Label inside bar
        if pct > 4:
            ax.text(left + v / 2, 0, f'{name}\n{v}例\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if i >= 3 else '#333')
        left += v

    # Arrow annotations showing progression
    ax.annotate('', xy=(total * 1.02, 0), xytext=(-total * 0.02, 0),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))
    ax.text(total * 0.5, -0.55, '← 探索                                                成熟 →',
            ha='center', va='center', fontsize=10, color='#666', fontstyle='italic')

    ax.set_xlim(-total * 0.03, total * 1.05)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title('产业应用成熟度分布', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    fig.subplots_adjust(top=0.88, bottom=0.10)
    _add_caption(fig, '图11 产业应用成熟度分布', y=0.01)
    _add_analysis(fig, '多数案例处于探索期和初期应用阶段，规模化和深度融合案例占比较低，产业整体仍处于早期。', x=0.05, y=0.10)
    _save(fig, 'fig11_industry_maturity.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 12 — Self-Developed Ratio (Waffle Chart)
# ═══════════════════════════════════════════════════════════════════════
def fig12():
    print('Fig12: Self-developed ratio waffle...')
    pct = selfdev_data['self_developed_ratio_pct']
    n_self = int(round(pct))  # ~14 out of 100
    n_third = 100 - n_self

    fig, ax = plt.subplots(figsize=(10, 8))

    # 10x10 grid of squares
    for i in range(100):
        row = i // 10
        col = i % 10
        is_self = i < n_self
        color = ACCENT if is_self else '#D5DBDB'
        rect = plt.Rectangle((col * 1.1, (9 - row) * 1.1), 1, 1,
                              facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.5, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Percentage callout
    ax.text(5.5, 11.5, f'自主研发比例: {pct:.1f}%', ha='center', va='center',
            fontsize=22, fontweight='bold', color=ACCENT)

    dist = selfdev_data['distribution']
    ax.text(5.5, -1.0,
            f'自主研发: {dist["TRUE"]}次  |  第三方采购: {dist["FALSE"]}次  |  总计: {dist["TRUE"]+dist["FALSE"]}次',
            ha='center', va='center', fontsize=11, color='#555')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=ACCENT, label=f'自主研发 ({n_self}%)'),
        mpatches.Patch(facecolor='#D5DBDB', label=f'第三方采购 ({n_third}%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              framealpha=0.9, bbox_to_anchor=(1.0, 1.0))

    ax.set_title('自主研发vs第三方采购比例', fontsize=16, fontweight='bold', color=PRIMARY, pad=30)
    fig.subplots_adjust(top=0.88, bottom=0.08)
    _add_caption(fig, '图12 自主研发vs第三方采购比例', y=0.01)
    _add_analysis(fig, '自主研发比例偏低，多数学校依赖第三方工具，自主创新能力有待提升。', x=0.05, y=0.08)
    _save(fig, 'fig12_self_developed_ratio.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 13 — Market Concentration (Lorenz Curve + Gini)
# ═══════════════════════════════════════════════════════════════════════
def fig13():
    print('Fig13: Market concentration Lorenz curve...')
    top20 = market_data['top20_companies']
    total_mentions = market_data['total_mentions']
    unique = market_data['unique_companies']
    cr5 = market_data['company_concentration_top5_pct']
    cr10 = market_data['company_concentration_top10_pct']

    # Build full distribution: top20 known + distribute rest
    known_vals = list(top20.values())
    remaining = total_mentions - sum(known_vals)
    n_remaining = unique - len(known_vals)
    if n_remaining > 0:
        avg_remaining = remaining / n_remaining
        all_vals = known_vals + [avg_remaining] * n_remaining
    else:
        all_vals = known_vals

    all_vals_sorted = sorted(all_vals)
    n = len(all_vals_sorted)
    cum_share = np.cumsum(all_vals_sorted) / sum(all_vals_sorted)
    cum_pop = np.arange(1, n + 1) / n

    # Gini coefficient
    gini = 1 - 2 * np.trapz(cum_share, cum_pop)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Equality line
    ax.plot([0, 1], [0, 1], '--', color='#999', linewidth=1.5, label='完全均等线')

    # Lorenz curve
    ax.plot(np.concatenate([[0], cum_pop]), np.concatenate([[0], cum_share]),
            color=PRIMARY, linewidth=2.5, label='洛伦兹曲线')

    # Shaded area
    ax.fill_between(np.concatenate([[0], cum_pop]),
                    np.concatenate([[0], cum_share]),
                    np.concatenate([[0], cum_pop]),
                    alpha=0.15, color=ACCENT)

    # CR5 / CR10 markers
    cr5_x = 1 - 5 / unique
    cr10_x = 1 - 10 / unique
    ax.axvline(cr5_x, color=WARM, linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axvline(cr10_x, color=TEAL, linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(cr5_x, 0.15, f'CR5={cr5:.1f}%', fontsize=9, color=WARM,
            ha='center', rotation=90)
    ax.text(cr10_x, 0.15, f'CR10={cr10:.1f}%', fontsize=9, color=TEAL,
            ha='center', rotation=90)

    # Gini annotation
    props = dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', edgecolor=WARM, alpha=0.95)
    ax.text(0.25, 0.85, f'Gini系数 = {gini:.3f}\n\n'
            f'CR5 = {cr5:.1f}%\nCR10 = {cr10:.1f}%\n'
            f'企业总数: {unique}\n提及总次数: {total_mentions}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=props, color='#333', linespacing=1.6)

    ax.set_xlabel('企业累计占比', fontsize=12, fontweight='bold')
    ax.set_ylabel('市场份额累计占比', fontsize=12, fontweight='bold')
    ax.set_title('企业市场集中度分析', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    _remove_spines(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图13 企业市场集中度分析（洛伦兹曲线）', y=0.01)
    _add_analysis(fig, '市场集中度较高，头部企业占据大部分份额，长尾效应显著，中小企业竞争激烈。', x=0.05, y=0.08)
    _save(fig, 'fig13_company_market_concentration.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 14 — LDA Topics (Horizontal Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════
def fig14():
    print('Fig14: LDA topics...')
    topics = lda_data['innovation']['topics']
    n_topics = len(topics)
    n_words = 8

    topic_labels = [
        '主题0: AI驱动精准教研',
        '主题1: 打破传统多模态融合',
        '主题2: 人机协同闭环',
        '主题3: 虚拟实验探究',
        '主题4: 深度融合范式',
        '主题5: 个性化精准批改',
        '主题6: 动态生成兴趣驱动',
    ]

    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 14), sharex=False)
    colors_topics = PALETTE_8[:n_topics]

    for i, (topic, ax) in enumerate(zip(topics, axes)):
        words = topic['top_words'][:n_words]
        w_names = [w['word'] for w in words][::-1]
        w_vals = [w['weight'] for w in words][::-1]

        ax.barh(range(n_words), w_vals, color=colors_topics[i], alpha=0.85,
                edgecolor='white', height=0.7)
        ax.set_yticks(range(n_words))
        ax.set_yticklabels(w_names, fontsize=8)
        _remove_spines(ax)

        label = topic_labels[i] if i < len(topic_labels) else f'主题{topic["topic_id"]}'
        ax.set_title(label, fontsize=10, fontweight='bold', color=colors_topics[i],
                     loc='left', pad=5)

        # Value labels
        for j, v in enumerate(w_vals):
            ax.text(v + 0.5, j, f'{v:.0f}', va='center', fontsize=7, color='#555')

    fig.suptitle('LDA主题模型：7大教学创新主题', fontsize=16, fontweight='bold',
                 color=PRIMARY, y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _add_caption(fig, '图14 LDA主题模型：7大教学创新主题', y=0.01)
    _add_analysis(fig, 'LDA主题建模识别出7个教学创新主题，涵盖精准教研、多模态融合、人机协同等方向。', x=0.05, y=0.06)
    _save(fig, 'fig14_lda_topics.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 15 — Tech Pathway Sankey (Alluvial Diagram)
# ═══════════════════════════════════════════════════════════════════════
def fig15():
    print('Fig15: Tech pathway sankey...')
    transitions = pathway_data['transitions']
    steps = pathway_data['step_frequency']

    # Build multi-layer flow: starting → middle → ending
    starting = {s['step']: s['count'] for s in pathway_data['starting_steps'][:8]}
    ending = {s['step']: s['count'] for s in pathway_data['ending_steps'][:8]}

    # Get top transitions
    top_trans = sorted(transitions, key=lambda x: -x['count'])[:25]

    # Assign layers: from_step → to_step
    all_from = set(t['from'] for t in top_trans)
    all_to = set(t['to'] for t in top_trans)
    left_nodes = list(all_from - all_to)[:6] + list(all_from & all_to)[:4]
    right_nodes = list(all_to - all_from)[:6] + list(all_from & all_to)[:4]

    # Deduplicate
    left_nodes = list(dict.fromkeys(left_nodes))[:8]
    right_nodes = list(dict.fromkeys(right_nodes))[:8]

    fig, ax = plt.subplots(figsize=(16, 10))

    # Position nodes
    n_left = len(left_nodes)
    n_right = len(right_nodes)
    left_y = np.linspace(0.9, 0.1, n_left)
    right_y = np.linspace(0.9, 0.1, n_right)

    left_pos = {n: (0.15, left_y[i]) for i, n in enumerate(left_nodes)}
    right_pos = {n: (0.85, right_y[i]) for i, n in enumerate(right_nodes)}

    # Draw connections
    max_count = max(t['count'] for t in top_trans)
    path_colors = plt.cm.get_cmap('RdYlBu_r')

    for t in top_trans:
        if t['from'] in left_pos and t['to'] in right_pos:
            x0, y0 = left_pos[t['from']]
            x1, y1 = right_pos[t['to']]
            alpha = 0.15 + 0.5 * t['count'] / max_count
            lw = 1 + 8 * t['count'] / max_count
            color = path_colors(t['count'] / max_count)

            # Bezier-like curve
            xs = np.linspace(x0, x1, 50)
            ys = y0 + (y1 - y0) * (3 * ((xs - x0) / (x1 - x0))**2 - 2 * ((xs - x0) / (x1 - x0))**3)
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')

    # Draw nodes
    for name, (x, y) in left_pos.items():
        ax.scatter(x, y, s=200, color=PRIMARY, zorder=5, edgecolors='white', linewidths=1.5)
        ax.text(x - 0.02, y, name, ha='right', va='center', fontsize=9, fontweight='bold', color='#333')

    for name, (x, y) in right_pos.items():
        ax.scatter(x, y, s=200, color=ACCENT, zorder=5, edgecolors='white', linewidths=1.5)
        ax.text(x + 0.02, y, name, ha='left', va='center', fontsize=9, fontweight='bold', color='#333')

    # Stage labels
    ax.text(0.15, 0.98, '起始阶段', ha='center', fontsize=12, fontweight='bold', color=PRIMARY)
    ax.text(0.85, 0.98, '终止阶段', ha='center', fontsize=12, fontweight='bold', color=ACCENT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis('off')
    ax.set_title('技术实施路径桑基图', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图15 技术实施路径桑基图', y=0.01)
    _add_analysis(fig, '技术路径呈现从"数据采集"到"智能反馈"的典型实施链条，路径多样性反映应用场景差异。', x=0.05, y=0.08)
    _save(fig, 'fig15_tech_pathway_sankey.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 16 — Scenario × Stage Bubble Matrix
# ═══════════════════════════════════════════════════════════════════════
def fig16():
    print('Fig16: Scenario × Stage bubble...')
    # Build from CSV
    stages_want = ['幼儿园', '小学', '初中', '高中']
    # Top scenarios from L2
    l2 = scenario_data['scenario_level2']
    top_scenarios = [k for k, v in sorted(l2.items(), key=lambda x: -x[1])
                     if k != '未提及'][:10]

    # Cross-tabulate
    matrix = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        st = str(row.get('学段', ''))
        sc = str(row.get('应用场景（二级）', ''))
        if st in stages_want and sc in top_scenarios:
            matrix[sc][st] += 1

    fig, ax = plt.subplots(figsize=(12, 9))

    x_labels = stages_want
    y_labels = top_scenarios[::-1]

    all_vals = []
    for sc in y_labels:
        for st in x_labels:
            all_vals.append(matrix[sc][st])
    max_val = max(all_vals) if all_vals else 1

    cmap = plt.cm.get_cmap('YlOrRd')

    for i, sc in enumerate(y_labels):
        for j, st in enumerate(x_labels):
            v = matrix[sc][st]
            if v > 0:
                size = 100 + 1500 * (v / max_val)
                color = cmap(v / max_val)
                ax.scatter(j, i, s=size, c=[color], alpha=0.75,
                           edgecolors='white', linewidths=1)
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#333')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    _remove_spines(ax)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)

    # Annotate top 5 bubbles
    top5 = sorted([(matrix[sc][st], sc, st) for sc in top_scenarios for st in stages_want],
                  reverse=True)[:5]
    for v, sc, st in top5:
        xi = x_labels.index(st)
        yi = y_labels.index(sc) if sc in y_labels else -1
        if yi >= 0:
            ax.annotate(f'{sc}×{st}\n{v}例',
                        xy=(xi, yi), xytext=(xi + 0.4, yi + 0.4),
                        fontsize=7, color=ACCENT,
                        arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.8))

    ax.set_title('应用场景×学段气泡矩阵', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)
    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图16 应用场景×学段气泡矩阵', y=0.01)
    _add_analysis(fig, '气泡矩阵揭示不同学段的场景偏好差异，小学阶段场景覆盖最广，高中阶段聚焦助学助评。', x=0.05, y=0.08)
    _save(fig, 'fig16_scenario_stage_bubble.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG 18 — Comprehensive Dashboard (2x3 grid)
# ═══════════════════════════════════════════════════════════════════════
def fig18():
    print('Fig18: Comprehensive dashboard...')
    fig = plt.figure(figsize=(20, 14))

    # Header banner
    fig.patches.append(FancyBboxPatch(
        (0.02, 0.92), 0.96, 0.07, boxstyle='round,pad=0.01',
        facecolor=PRIMARY, edgecolor='none', transform=fig.transFigure,
        zorder=10))
    fig.text(0.5, 0.955, 'AI赋能基础教育综合仪表盘', ha='center', va='center',
             fontsize=22, fontweight='bold', color='white', zorder=11)

    # Key stats
    total_cases = 1690
    total_tools = tool_data['total_tool_mentions']
    n_provinces = len([k for k in province_data['case_count_by_province'] if k != '未提及'])
    stats_text = f'案例总数: {total_cases}  |  工具提及: {total_tools}次  |  覆盖省份: {n_provinces}个  |  自研比例: {selfdev_data["self_developed_ratio_pct"]:.1f}%'
    fig.text(0.5, 0.915, stats_text, ha='center', va='center',
             fontsize=11, color='#666', zorder=11)

    # ── Panel 1: Donut (stage) ──
    ax1 = fig.add_subplot(2, 3, 1)
    raw = stage_data['case_count_by_stage']
    main = {'小学': raw.get('小学', 0), '初中': raw.get('初中', 0),
            '高中': raw.get('高中', 0), '幼儿园': raw.get('幼儿园', 0)}
    other = sum(v for k, v in raw.items() if k not in main)
    if other > 0:
        main['其他'] = other
    colors_d = [PRIMARY, SECONDARY, TEAL, WARM, '#BDC3C7']
    wedges, _ = ax1.pie(main.values(), colors=colors_d[:len(main)],
                        wedgeprops=dict(width=0.4, edgecolor='white'), startangle=90)
    ax1.text(0, 0, f'N={total_cases}', ha='center', va='center',
             fontsize=12, fontweight='bold', color=PRIMARY)
    ax1.set_title('学段分布', fontsize=11, fontweight='bold', color=PRIMARY)

    # ── Panel 2: Top 10 tools bar ──
    ax2 = fig.add_subplot(2, 3, 2)
    top10 = list(tool_data['top30_tools'].items())[:10]
    names = [x[0] for x in top10][::-1]
    vals = [x[1] for x in top10][::-1]
    ax2.barh(range(len(names)), vals, color=SECONDARY, alpha=0.85, height=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7)
    _remove_spines(ax2)
    ax2.set_title('Top10工具', fontsize=11, fontweight='bold', color=PRIMARY)

    # ── Panel 3: Radar (五育) ──
    ax3 = fig.add_subplot(2, 3, 3, polar=True)
    wuyu = ['智育', '德育', '美育', '体育', '劳育']
    cult = scenario_data['cultivation_direction']
    cult_vals = []
    for w in wuyu:
        total_w = sum(v for k, v in cult.items() if w in k)
        cult_vals.append(total_w)
    mx = max(cult_vals) if cult_vals else 1
    cult_norm = [v / mx for v in cult_vals]
    angles = np.linspace(0, 2 * np.pi, len(wuyu), endpoint=False).tolist()
    angles += angles[:1]
    cult_norm += cult_norm[:1]
    ax3.plot(angles, cult_norm, 'o-', color=TEAL, linewidth=2)
    ax3.fill(angles, cult_norm, alpha=0.15, color=TEAL)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(wuyu, fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.set_title('五育融合', fontsize=11, fontweight='bold', color=PRIMARY, pad=15)

    # ── Panel 4: Mini heatmap ──
    ax4 = fig.add_subplot(2, 3, 4)
    ct = cross_data['stage_subject']
    main_stages = ['小学', '初中', '高中', '幼儿园']
    stage_idx = [ct['index'].index(s) for s in main_stages if s in ct['index']]
    subj_idx = list(range(min(8, len(ct['columns']))))
    filtered = np.array(ct['data'])[np.ix_(stage_idx, subj_idx)]
    cmap_h = mcolors.LinearSegmentedColormap.from_list('bw', ['#EBF5FB', PRIMARY])
    ax4.imshow(filtered, cmap=cmap_h, aspect='auto')
    ax4.set_xticks(range(len(subj_idx)))
    ax4.set_xticklabels([ct['columns'][i] for i in subj_idx], fontsize=6, rotation=45, ha='right')
    ax4.set_yticks(range(len(stage_idx)))
    ax4.set_yticklabels([ct['index'][i] for i in stage_idx], fontsize=7)
    ax4.set_title('学段×学科', fontsize=11, fontweight='bold', color=PRIMARY)
    ax4.spines[:].set_visible(False)

    # ── Panel 5: Mini network ──
    ax5 = fig.add_subplot(2, 3, 5)
    net = cooccurrence_data['network']
    G = nx.Graph()
    for n in net['nodes'][:20]:
        G.add_node(n['id'], freq=n['freq'])
    for e in net['edges'][:30]:
        if e['source'] in G.nodes() and e['target'] in G.nodes():
            G.add_edge(e['source'], e['target'], weight=e['weight'])
    pos = nx.spring_layout(G, k=2, seed=42)
    deg = nx.degree_centrality(G)
    sizes = [100 + 500 * deg.get(n, 0) for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax5, alpha=0.15, edge_color='#999')
    nx.draw_networkx_nodes(G, pos, ax=ax5, node_size=sizes, node_color=SECONDARY,
                           alpha=0.7, edgecolors='white')
    top5_n = sorted(G.nodes(), key=lambda n: deg.get(n, 0), reverse=True)[:8]
    nx.draw_networkx_labels(G, pos, {n: n for n in top5_n}, ax=ax5, font_size=6)
    ax5.axis('off')
    ax5.set_title('共现网络', fontsize=11, fontweight='bold', color=PRIMARY)

    # ── Panel 6: Province top 10 ──
    ax6 = fig.add_subplot(2, 3, 6)
    prov = province_data['case_count_by_province']
    prov = {k: v for k, v in prov.items() if k != '未提及'}
    top10_p = sorted(prov.items(), key=lambda x: -x[1])[:10]
    p_names = [x[0] for x in top10_p][::-1]
    p_vals = [x[1] for x in top10_p][::-1]
    norm = plt.Normalize(min(p_vals), max(p_vals))
    cmap_g = mcolors.LinearSegmentedColormap.from_list('bg', ['#D6EAF8', PRIMARY])
    ax6.barh(range(len(p_names)), p_vals, color=[cmap_g(norm(v)) for v in p_vals], height=0.7)
    ax6.set_yticks(range(len(p_names)))
    ax6.set_yticklabels(p_names, fontsize=7)
    _remove_spines(ax6)
    ax6.set_title('Top10省份', fontsize=11, fontweight='bold', color=PRIMARY)

    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.3, bottom=0.06)
    _add_caption(fig, '图18 AI赋能基础教育综合仪表盘', y=0.01)
    _save(fig, 'fig18_comprehensive_dashboard.png')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('Premium Visualization Redesign')
    print('='*60)

    funcs = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08,
             fig09, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig18]

    for fn in funcs:
        try:
            fn()
        except Exception as e:
            print(f'  [ERROR] {fn.__name__}: {e}')
            import traceback
            traceback.print_exc()

    print('='*60)
    print(f'Done. Figures saved to {FIG_DIR}')
    print('='*60)
