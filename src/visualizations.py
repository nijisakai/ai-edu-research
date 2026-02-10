#!/usr/bin/env python3
"""
AI Education Industry Research - Comprehensive Visualization Suite
Generates 18+ publication-quality figures for the research report.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from collections import Counter, defaultdict
import math

warnings.filterwarnings('ignore')

# ── Global Configuration ──────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

BASE_DIR = '/Users/sakai/Desktop/产业调研/ai-edu-research'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
CSV_PATH = '/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv'

os.makedirs(FIG_DIR, exist_ok=True)

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#1B4F72',
    'secondary': '#2E86C1',
    'accent1': '#17A589',
    'accent2': '#E67E22',
    'accent3': '#C0392B',
    'accent4': '#8E44AD',
    'light_blue': '#AED6F1',
    'light_teal': '#A3E4D7',
    'light_orange': '#F5CBA7',
    'bg_gray': '#F8F9FA',
}

PALETTE_MAIN = ['#1B4F72', '#2E86C1', '#17A589', '#E67E22', '#C0392B',
                '#8E44AD', '#2C3E50', '#16A085', '#D35400', '#7D3C98']
PALETTE_EXTENDED = PALETTE_MAIN + [
    '#1ABC9C', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12',
    '#27AE60', '#2980B9', '#8E44AD', '#D35400', '#C0392B',
    '#1F618D', '#148F77', '#B7950B', '#A04000', '#6C3483']

GRADIENT_BLUES = LinearSegmentedColormap.from_list('custom_blues',
    ['#EBF5FB', '#AED6F1', '#5DADE2', '#2E86C1', '#1B4F72'])
GRADIENT_TEAL_ORANGE = LinearSegmentedColormap.from_list('teal_orange',
    ['#A3E4D7', '#17A589', '#F5CBA7', '#E67E22'])


# ── Data Loading Helpers ──────────────────────────────────────────────────────
def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

def load_csv():
    return pd.read_csv(CSV_PATH, encoding='utf-8-sig')

def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}.png / .pdf')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01: Province Distribution Heatmap (Horizontal Bar with Gradient)
# ══════════════════════════════════════════════════════════════════════════════
def fig01_province_heatmap():
    print('Creating FIG 01: Province distribution...')
    data = load_json('province_distribution.json')
    prov = data['case_count_by_province']
    # Remove '未提及'
    prov = {k: v for k, v in prov.items() if k != '未提及'}
    # Sort ascending for horizontal bar
    sorted_items = sorted(prov.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 12))
    norm = Normalize(vmin=min(values), vmax=max(values))
    colors = [GRADIENT_BLUES(norm(v)) for v in values]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('案例数量', fontsize=12)
    ax.set_title('全国各省份AI教育案例分布', fontsize=16, fontweight='bold', pad=20)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values)*0.01, i, str(val), va='center', fontsize=8, color='#333')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=GRADIENT_BLUES, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.3, aspect=15, pad=0.02)
    cbar.set_label('案例数量', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    fig.tight_layout()
    save_fig(fig, 'fig01_province_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02: School Stage Donut Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig02_school_stage_donut():
    print('Creating FIG 02: School stage donut...')
    data = load_json('stage_distribution.json')
    stage = data['case_count_by_stage']
    # Keep main stages, group small ones
    main_stages = ['小学', '初中', '高中', '幼儿园', '中职', '大学']
    main_vals = {}
    other = 0
    for k, v in stage.items():
        if k in main_stages:
            main_vals[k] = v
        else:
            other += v
    if other > 0:
        main_vals['其他'] = other

    labels = list(main_vals.keys())
    sizes = list(main_vals.values())
    total = sum(sizes)
    colors = PALETTE_EXTENDED[:len(labels)]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='', startangle=90,
        colors=colors, pctdistance=0.82,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))

    # Add custom labels with count and percentage
    for i, (wedge, label, size) in enumerate(zip(wedges, labels, sizes)):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        pct = size / total * 100
        ax.annotate(f'{label}\n{size} ({pct:.1f}%)',
                    xy=(x * 0.8, y * 0.8),
                    xytext=(x * 1.35, y * 1.35),
                    fontsize=10, ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.15))

    # Center text
    ax.text(0, 0, f'总计\n{total}', ha='center', va='center',
            fontsize=18, fontweight='bold', color=COLORS['primary'])
    ax.set_title('学段分布', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    save_fig(fig, 'fig02_school_stage_donut')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03: Top 20 AI Tools Horizontal Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig03_top20_tools_bar():
    print('Creating FIG 03: Top 20 AI tools...')
    data = load_json('tool_product_distribution.json')
    tools = data['top30_tools']
    # Take top 20
    sorted_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)[:20]
    sorted_tools.reverse()  # For horizontal bar (bottom to top)
    names = [x[0] for x in sorted_tools]
    values = [x[1] for x in sorted_tools]

    # Company color mapping
    company_colors = {
        '豆包': '#3B82F6', 'DeepSeek': '#10B981', '即梦AI': '#6366F1',
        '即梦 AI': '#6366F1', '剪映AI': '#EC4899', '希沃白板': '#F59E0B',
        'Kimi': '#8B5CF6', '通义千问': '#EF4444', '文心一言': '#3B82F6',
        '讯飞星火': '#14B8A6', 'ChatGPT': '#10B981', '可画': '#F97316',
    }
    colors = []
    for name in names:
        matched = False
        for key, color in company_colors.items():
            if key in name:
                colors.append(color)
                matched = True
                break
        if not matched:
            colors.append(COLORS['secondary'])

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(names)), values, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('使用次数', fontsize=12)
    ax.set_title('Top 20 AI教育工具/产品使用频次', fontsize=16, fontweight='bold', pad=20)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + max(values)*0.01, i, str(val), va='center', fontsize=9, color='#333')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save_fig(fig, 'fig03_top20_tools_bar')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04: Product Ecosystem Treemap
# ══════════════════════════════════════════════════════════════════════════════
def fig04_product_ecosystem_treemap():
    print('Creating FIG 04: Product ecosystem treemap...')
    import squarify
    data = load_json('tool_product_distribution.json')
    form_data = data['product_form']
    cat_data = data['product_category']

    # Treemap of product categories
    sorted_cats = sorted(cat_data.items(), key=lambda x: x[1], reverse=True)
    # Keep top 10, group rest
    top_cats = sorted_cats[:10]
    other_val = sum(v for _, v in sorted_cats[10:])
    if other_val > 0:
        top_cats.append(('其他', other_val))

    labels = [f'{k}\n({v})' for k, v in top_cats]
    sizes = [v for _, v in top_cats]
    colors = PALETTE_EXTENDED[:len(sizes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Product category treemap
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85,
                  ax=ax1, text_kwargs={'fontsize': 9, 'fontweight': 'bold'})
    ax1.set_title('产品分类分布', fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')

    # Right: Product form treemap
    sorted_forms = sorted(form_data.items(), key=lambda x: x[1], reverse=True)
    top_forms = sorted_forms[:8]
    other_form = sum(v for _, v in sorted_forms[8:])
    if other_form > 0:
        top_forms.append(('其他', other_form))
    f_labels = [f'{k}\n({v})' for k, v in top_forms]
    f_sizes = [v for _, v in top_forms]
    f_colors = [COLORS['primary'], COLORS['accent1'], COLORS['accent2'],
                COLORS['accent3'], COLORS['accent4'], COLORS['secondary'],
                '#1ABC9C', '#F39C12', '#95A5A6'][:len(f_sizes)]

    squarify.plot(sizes=f_sizes, label=f_labels, color=f_colors, alpha=0.85,
                  ax=ax2, text_kwargs={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('产品形态分布', fontsize=14, fontweight='bold', pad=15)
    ax2.axis('off')

    fig.suptitle('AI教育产品生态图谱', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig04_product_ecosystem_treemap')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05: Scenario Sunburst (Nested Pie)
# ══════════════════════════════════════════════════════════════════════════════
def fig05_scenario_sunburst():
    print('Creating FIG 05: Scenario sunburst...')
    data = load_json('scenario_analysis.json')
    l1 = data['scenario_level1']
    l2 = data['scenario_level2']

    # Remove '未提及' from L1
    l1 = {k: v for k, v in l1.items() if k != '未提及'}

    # Map L2 to L1 using the CSV
    df = load_csv()
    l2_to_l1 = {}
    for _, row in df.iterrows():
        s_l1 = str(row.get('应用场景（一级）', ''))
        s_l2 = str(row.get('应用场景（二级）', ''))
        if s_l1 in l1 and s_l2 and s_l2 != 'nan':
            if s_l2 not in l2_to_l1:
                l2_to_l1[s_l2] = Counter()
            l2_to_l1[s_l2][s_l1] += 1

    # Build nested structure: for each L1, get top L2 subcategories
    l1_colors = {
        '助学': '#2E86C1', '助教': '#17A589', '助育': '#E67E22',
        '助评': '#C0392B', '助管': '#8E44AD', '助研': '#2C3E50'
    }

    fig, ax = plt.subplots(figsize=(12, 12))

    # Outer ring: L1
    l1_sorted = sorted(l1.items(), key=lambda x: x[1], reverse=True)
    l1_labels = [x[0] for x in l1_sorted]
    l1_sizes = [x[1] for x in l1_sorted]
    l1_cols = [l1_colors.get(k, '#95A5A6') for k in l1_labels]

    # Inner ring: L1 (same data, smaller)
    wedges1, texts1 = ax.pie(l1_sizes, radius=0.65, colors=l1_cols,
                              wedgeprops=dict(width=0.25, edgecolor='white', linewidth=2),
                              startangle=90, labels=None)

    # Build L2 data aligned with L1 sectors
    l2_sizes_all = []
    l2_colors_all = []
    l2_labels_all = []
    for l1_name in l1_labels:
        # Find L2 items belonging to this L1
        l2_for_l1 = {}
        for l2_name, l1_counter in l2_to_l1.items():
            if l1_counter.most_common(1)[0][0] == l1_name:
                l2_for_l1[l2_name] = l2.get(l2_name, 0)
        # Sort and take top 4
        l2_sorted = sorted(l2_for_l1.items(), key=lambda x: x[1], reverse=True)[:4]
        l2_sum = sum(v for _, v in l2_sorted)
        l1_total = l1[l1_name]
        remainder = l1_total - l2_sum
        base_color = l1_colors.get(l1_name, '#95A5A6')
        # Create shades
        from matplotlib.colors import to_rgba
        base_rgba = to_rgba(base_color)
        for j, (l2n, l2v) in enumerate(l2_sorted):
            alpha = 0.9 - j * 0.15
            c = (*base_rgba[:3], alpha)
            l2_sizes_all.append(l2v)
            l2_colors_all.append(c)
            l2_labels_all.append(l2n)
        if remainder > 0:
            l2_sizes_all.append(remainder)
            l2_colors_all.append((*base_rgba[:3], 0.3))
            l2_labels_all.append('')

    wedges2, texts2 = ax.pie(l2_sizes_all, radius=1.0, colors=l2_colors_all,
                              wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1),
                              startangle=90, labels=None)

    # Add L1 labels
    for i, (wedge, label, size) in enumerate(zip(wedges1, l1_labels, l1_sizes)):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = np.cos(np.deg2rad(ang)) * 0.52
        y = np.sin(np.deg2rad(ang)) * 0.52
        pct = size / sum(l1_sizes) * 100
        ax.text(x, y, f'{label}\n{pct:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    # Legend for L2 (top items)
    top_l2 = sorted(l2.items(), key=lambda x: x[1], reverse=True)[:10]
    legend_text = '\n'.join([f'{k}: {v}' for k, v in top_l2])
    ax.text(1.15, -0.3, 'Top 二级场景:\n' + legend_text,
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))

    ax.set_title('应用场景层级分布（一级→二级）', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    save_fig(fig, 'fig05_scenario_sunburst')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06: Subject Distribution (Polar Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig06_subject_distribution():
    print('Creating FIG 06: Subject distribution...')
    data = load_json('subject_distribution.json')
    subj = data['top20_subjects']
    # Take top 15 subjects, exclude '未提及'
    sorted_subj = sorted(subj.items(), key=lambda x: x[1], reverse=True)
    sorted_subj = [(k, v) for k, v in sorted_subj if k != '未提及'][:15]

    labels = [x[0] for x in sorted_subj]
    values = [x[1] for x in sorted_subj]
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Bar chart on polar axes
    width = 2 * np.pi / N * 0.75
    norm = Normalize(vmin=min(values), vmax=max(values))
    colors = [GRADIENT_BLUES(norm(v)) for v in values]

    bars = ax.bar(angles, values, width=width, color=colors,
                  edgecolor='white', linewidth=1.5, alpha=0.85)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('学科分布（Top 15）', fontsize=16, fontweight='bold', pad=30)

    # Add value labels
    for angle, val in zip(angles, values):
        ax.text(angle, val + max(values)*0.05, str(val),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylim(0, max(values) * 1.15)
    ax.spines['polar'].set_visible(False)
    fig.tight_layout()
    save_fig(fig, 'fig06_subject_distribution')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07: Cultivation Radar (五育 by School Stage)
# ══════════════════════════════════════════════════════════════════════════════
def fig07_cultivation_radar():
    print('Creating FIG 07: Cultivation radar...')
    data = load_json('nlp_cultivation.json')
    by_stage = data['by_school_level']

    wuyu = ['智育', '德育', '美育', '体育', '劳育']
    stages_to_plot = ['小学', '初中', '高中', '幼儿园']
    stage_colors = ['#2E86C1', '#17A589', '#E67E22', '#C0392B']

    N = len(wuyu)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for stage, color in zip(stages_to_plot, stage_colors):
        if stage not in by_stage:
            continue
        stage_data = by_stage[stage]
        vals = [stage_data.get(w, 0) for w in wuyu]
        # Normalize to percentage within stage
        total = sum(stage_data.values()) if sum(stage_data.values()) > 0 else 1
        vals_pct = [v / total * 100 for v in vals]
        vals_pct += vals_pct[:1]
        ax.plot(angles, vals_pct, 'o-', linewidth=2, label=stage, color=color, markersize=6)
        ax.fill(angles, vals_pct, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(wuyu, fontsize=13, fontweight='bold')
    ax.set_title('五育培养方向雷达图（按学段）', fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_ylim(0, None)
    fig.tight_layout()
    save_fig(fig, 'fig07_cultivation_radar')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08: Technology Word Cloud
# ══════════════════════════════════════════════════════════════════════════════
def fig08_tech_wordcloud():
    print('Creating FIG 08: Technology word cloud...')
    from wordcloud import WordCloud
    import jieba

    data = load_json('nlp_tfidf_keywords.json')
    # Combine keywords from innovation and tech path
    word_freq = Counter()
    for field in ['innovation_case', 'tech_path_case', 'industry_status']:
        if field in data and 'top_keywords' in data[field]:
            for item in data[field]['top_keywords']:
                word_freq[item['word']] += int(item.get('doc_freq', item.get('weight', 1)))

    # Also add from co-occurrence keywords
    cooc = load_json('nlp_cooccurrence.json')
    for item in cooc['top_keywords']:
        word_freq[item['word']] += item['freq']

    # Filter out single chars and common stopwords
    stopwords = {'的', '了', '在', '是', '和', '与', '对', '为', '等', '中', '将', '能',
                 '通过', '进行', '实现', '使用', '利用', '基于', '可以', '以及', '不同',
                 '提供', '支持', '相关', '具有', '采用', '结合', '方面', '过程', '方式'}
    word_freq = {k: v for k, v in word_freq.items()
                 if len(k) >= 2 and k not in stopwords}

    # Try to find a font
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    font_path = None
    for fp in font_paths:
        if os.path.exists(fp):
            font_path = fp
            break

    wc = WordCloud(
        font_path=font_path,
        width=1600, height=900,
        background_color='white',
        max_words=150,
        max_font_size=120,
        min_font_size=10,
        colormap='ocean',
        prefer_horizontal=0.7,
        relative_scaling=0.5,
    )
    wc.generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('AI教育技术关键词云', fontsize=18, fontweight='bold', pad=20)
    fig.tight_layout()
    save_fig(fig, 'fig08_tech_wordcloud')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09: Technology Co-occurrence Network
# ══════════════════════════════════════════════════════════════════════════════
def fig09_tech_cooccurrence_network():
    print('Creating FIG 09: Co-occurrence network...')
    import networkx as nx

    data = load_json('nlp_cooccurrence.json')
    network = data['network']

    G = nx.Graph()
    for node in network['nodes']:
        G.add_node(node['id'], freq=node['freq'])
    for edge in network['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    fig, ax = plt.subplots(figsize=(14, 14))

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Node sizes based on frequency
    freqs = [G.nodes[n].get('freq', 10) for n in G.nodes()]
    max_freq = max(freqs) if freqs else 1
    node_sizes = [300 + (f / max_freq) * 2500 for f in freqs]

    # Edge widths based on weight
    weights = [G.edges[e]['weight'] for e in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + (w / max_w) * 4 for w in weights]
    edge_alphas = [0.2 + (w / max_w) * 0.6 for w in weights]

    # Draw edges
    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                               alpha=alpha, edge_color=COLORS['secondary'], ax=ax)

    # Node colors by degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    norm = Normalize(vmin=0, vmax=max_deg)
    node_colors = [GRADIENT_BLUES(norm(degrees[n])) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors='white', linewidths=1.5, ax=ax)

    # Labels
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9,
                            font_family='Arial Unicode MS', ax=ax)

    ax.set_title('AI教育技术关键词共现网络', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add legend for node size
    ax.text(0.02, 0.02, f'节点大小 = 词频\n连线粗细 = 共现频次\n节点数: {len(G.nodes())}\n连线数: {len(G.edges())}',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))

    fig.tight_layout()
    save_fig(fig, 'fig09_tech_cooccurrence_network')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10: Stage x Subject Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig10_stage_subject_heatmap():
    print('Creating FIG 10: Stage x Subject heatmap...')
    data = load_json('cross_tabulations.json')
    ct = data['stage_subject']
    df = pd.DataFrame(ct['data'], index=ct['index'], columns=ct['columns'])

    # Filter to main stages and top subjects
    main_stages = ['小学', '初中', '高中', '幼儿园', '中职']
    main_stages = [s for s in main_stages if s in df.index]
    top_subjects = df.loc[main_stages].sum().sort_values(ascending=False).head(12).index.tolist()
    top_subjects = [s for s in top_subjects if s != '未提及']

    df_plot = df.loc[main_stages, top_subjects]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(df_plot.values, cmap=GRADIENT_BLUES, aspect='auto')

    ax.set_xticks(range(len(top_subjects)))
    ax.set_xticklabels(top_subjects, fontsize=11, rotation=45, ha='right')
    ax.set_yticks(range(len(main_stages)))
    ax.set_yticklabels(main_stages, fontsize=12)

    # Add text annotations
    for i in range(len(main_stages)):
        for j in range(len(top_subjects)):
            val = df_plot.values[i, j]
            color = 'white' if val > df_plot.values.max() * 0.6 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('案例数量', fontsize=11)
    ax.set_title('学段 × 学科 交叉分布热力图', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    save_fig(fig, 'fig10_stage_subject_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11: Industry Maturity Distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig11_industry_maturity():
    print('Creating FIG 11: Industry maturity...')
    data = load_json('industry_maturity.json')
    maturity = data['maturity_keyword_freq']

    # Group into maturity stages
    stage_map = {
        '萌芽期': ['初期', '起步', '萌芽', '早期'],
        '探索期': ['探索', '试点', '尝试'],
        '发展期': ['发展', '成长', '推广', '普及'],
        '成熟期': ['成熟', '深化', '规模化'],
    }
    stage_vals = {}
    used_keys = set()
    for stage_name, keywords in stage_map.items():
        total = 0
        for kw in keywords:
            if kw in maturity:
                total += maturity[kw]
                used_keys.add(kw)
        stage_vals[stage_name] = total

    # Add remaining
    other = sum(v for k, v in maturity.items() if k not in used_keys)
    if other > 0:
        stage_vals['其他'] = other

    labels = list(stage_vals.keys())
    values = list(stage_vals.values())
    colors = ['#AED6F1', '#5DADE2', '#2E86C1', '#1B4F72', '#95A5A6'][:len(labels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.2]})

    # Left: Stacked horizontal bar (gauge-like)
    cumulative = 0
    total = sum(values)
    for i, (label, val) in enumerate(zip(labels, values)):
        width = val / total
        ax1.barh(0, width, left=cumulative, color=colors[i], edgecolor='white',
                 height=0.5, label=f'{label} ({val})')
        if width > 0.05:
            ax1.text(cumulative + width/2, 0, f'{label}\n{val/total*100:.0f}%',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     color='white' if i >= 2 else 'black')
        cumulative += width

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axis('off')
    ax1.set_title('产业成熟度分布', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=9)

    # Right: Detailed keyword frequency
    sorted_kw = sorted(maturity.items(), key=lambda x: x[1], reverse=True)[:15]
    kw_names = [x[0] for x in sorted_kw][::-1]
    kw_vals = [x[1] for x in sorted_kw][::-1]
    norm = Normalize(vmin=min(kw_vals), vmax=max(kw_vals))
    kw_colors = [GRADIENT_BLUES(norm(v)) for v in kw_vals]

    ax2.barh(range(len(kw_names)), kw_vals, color=kw_colors, edgecolor='white')
    ax2.set_yticks(range(len(kw_names)))
    ax2.set_yticklabels(kw_names, fontsize=10)
    ax2.set_xlabel('出现频次', fontsize=11)
    ax2.set_title('产业成熟度关键词频次', fontsize=14, fontweight='bold')
    for i, v in enumerate(kw_vals):
        ax2.text(v + max(kw_vals)*0.01, i, str(v), va='center', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('AI教育产业成熟度分析', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig11_industry_maturity')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12: Self-Developed vs Third-Party
# ══════════════════════════════════════════════════════════════════════════════
def fig12_self_developed_ratio():
    print('Creating FIG 12: Self-developed ratio...')
    data = load_json('self_developed_ratio.json')
    dist = data['distribution']
    ratio = data['self_developed_ratio_pct']

    # Map TRUE/FALSE to labels
    label_map = {'TRUE': '自主研发', 'FALSE': '第三方工具'}
    labels = [label_map.get(k, k) for k in dist.keys()]
    values = list(dist.values())
    colors = [COLORS['accent1'], COLORS['secondary']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Donut
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
        textprops={'fontsize': 12})
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax1.text(0, 0, f'{ratio:.1f}%\n自研率', ha='center', va='center',
             fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax1.set_title('自主研发 vs 第三方工具', fontsize=14, fontweight='bold')

    # Right: Breakdown by school stage from CSV
    df = load_csv()
    stage_self = df.groupby(['学段', '是否自主研发']).size().unstack(fill_value=0)
    main_stages = ['小学', '初中', '高中', '幼儿园']
    stage_self = stage_self.reindex([s for s in main_stages if s in stage_self.index])

    if True in stage_self.columns or 'TRUE' in stage_self.columns:
        true_col = True if True in stage_self.columns else 'TRUE'
        false_col = False if False in stage_self.columns else 'FALSE'
    else:
        true_col = stage_self.columns[0]
        false_col = stage_self.columns[1] if len(stage_self.columns) > 1 else stage_self.columns[0]

    x = range(len(stage_self.index))
    width = 0.35
    ax2.bar([i - width/2 for i in x], stage_self[false_col], width,
            label='第三方工具', color=COLORS['secondary'], edgecolor='white')
    ax2.bar([i + width/2 for i in x], stage_self[true_col], width,
            label='自主研发', color=COLORS['accent1'], edgecolor='white')

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(stage_self.index, fontsize=11)
    ax2.set_ylabel('工具数量', fontsize=11)
    ax2.set_title('各学段自研比例对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('AI教育工具自主研发分析', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig12_self_developed_ratio')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13: Market Concentration (Lorenz Curve)
# ══════════════════════════════════════════════════════════════════════════════
def fig13_company_market_concentration():
    print('Creating FIG 13: Market concentration...')
    data = load_json('company_market_share.json')
    top20 = data['top20_companies']
    top5_pct = data['company_concentration_top5_pct']
    top10_pct = data['company_concentration_top10_pct']
    total_mentions = data['total_mentions']
    unique_companies = data['unique_companies']

    # Build full distribution for Lorenz curve
    df = load_csv()
    company_counts = df['工具标准名'].value_counts()
    sorted_counts = np.sort(company_counts.values)
    cumulative = np.cumsum(sorted_counts) / sorted_counts.sum()
    x_lorenz = np.arange(1, len(cumulative) + 1) / len(cumulative)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Lorenz curve
    ax1.plot(x_lorenz, cumulative, color=COLORS['primary'], linewidth=2.5, label='实际分布')
    ax1.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='完全均匀分布')
    ax1.fill_between(x_lorenz, cumulative, x_lorenz[:len(cumulative)],
                     alpha=0.15, color=COLORS['primary'])

    # Calculate Gini coefficient
    n = len(sorted_counts)
    gini = (2 * np.sum(np.arange(1, n+1) * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n

    ax1.set_xlabel('工具/产品累计占比', fontsize=12)
    ax1.set_ylabel('市场份额累计占比', fontsize=12)
    ax1.set_title('市场集中度 - 洛伦兹曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.text(0.05, 0.85, f'基尼系数: {gini:.3f}\nTop5集中度: {top5_pct:.1f}%\nTop10集中度: {top10_pct:.1f}%',
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.5))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Top 20 companies bar
    sorted_top20 = sorted(top20.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_top20]
    vals = [x[1] for x in sorted_top20]
    norm = Normalize(vmin=min(vals), vmax=max(vals))
    bar_colors = [GRADIENT_BLUES(norm(v)) for v in vals]

    ax2.barh(range(len(names)), vals, color=bar_colors, edgecolor='white')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('使用次数', fontsize=11)
    ax2.set_title('Top 20 工具/产品市场份额', fontsize=14, fontweight='bold')
    for i, v in enumerate(vals):
        pct = v / total_mentions * 100
        ax2.text(v + max(vals)*0.01, i, f'{v} ({pct:.1f}%)', va='center', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('AI教育市场集中度分析', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig13_company_market_concentration')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 14: LDA Topic Model Visualization
# ══════════════════════════════════════════════════════════════════════════════
def fig14_lda_topics():
    print('Creating FIG 14: LDA topics...')
    data = load_json('nlp_lda_topics.json')
    topics = data['innovation']['topics']

    n_topics = len(topics)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    topic_labels = [
        'AI驱动精准教学', '打破传统壁垒', '人机协同育人', '个性化学习',
        '数据驱动评价', '智能创作工具', '跨学科融合', '技术赋能课堂'
    ]

    for i, topic in enumerate(topics[:8]):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
        words = topic['top_words'][:10]
        w_names = [w['word'] for w in words][::-1]
        w_weights = [w['weight'] for w in words][::-1]

        norm = Normalize(vmin=min(w_weights), vmax=max(w_weights))
        colors = [GRADIENT_BLUES(norm(w)) for w in w_weights]

        ax.barh(range(len(w_names)), w_weights, color=colors, edgecolor='white')
        ax.set_yticks(range(len(w_names)))
        ax.set_yticklabels(w_names, fontsize=9)
        label = topic_labels[i] if i < len(topic_labels) else f'Topic {i}'
        ax.set_title(f'Topic {i+1}: {label}', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)

    # Hide unused axes
    for j in range(min(n_topics, 8), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('LDA主题模型 - 创新点主题分布', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig14_lda_topics')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 15: Technology Pathway Sankey-style Diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig15_tech_pathway_sankey():
    print('Creating FIG 15: Tech pathway Sankey...')
    data = load_json('nlp_tech_pathways.json')
    transitions = data['transitions'][:20]  # Top 20 transitions
    starting = data['starting_steps'][:8]
    ending = data['ending_steps'][:8]

    # Build alluvial-style with matplotlib
    fig, ax = plt.subplots(figsize=(18, 10))

    # Collect all unique steps
    all_steps = set()
    for t in transitions:
        all_steps.add(t['from'])
        all_steps.add(t['to'])

    # Classify steps into columns (start, middle, end)
    start_names = set(s['step'] for s in starting)
    end_names = set(s['step'] for s in ending)
    mid_names = all_steps - start_names - end_names

    # If overlap, keep in start/end
    columns = [
        sorted(start_names & all_steps, key=lambda x: -sum(t['count'] for t in transitions if t['from']==x)),
        sorted(mid_names, key=lambda x: -sum(t['count'] for t in transitions if t['from']==x or t['to']==x)),
        sorted(end_names & all_steps, key=lambda x: -sum(t['count'] for t in transitions if t['to']==x)),
    ]

    # Assign positions
    col_x = [0.1, 0.5, 0.9]
    step_pos = {}
    for ci, col in enumerate(columns):
        n = len(col)
        for si, step in enumerate(col):
            y = 0.9 - (si / max(n-1, 1)) * 0.8 if n > 1 else 0.5
            step_pos[step] = (col_x[ci], y)

    # Draw connections
    max_count = max(t['count'] for t in transitions)
    for t in transitions:
        if t['from'] in step_pos and t['to'] in step_pos:
            x1, y1 = step_pos[t['from']]
            x2, y2 = step_pos[t['to']]
            alpha = 0.15 + (t['count'] / max_count) * 0.5
            lw = 1 + (t['count'] / max_count) * 8
            # Bezier-like curve
            mid_x = (x1 + x2) / 2
            from matplotlib.patches import FancyArrowPatch
            from matplotlib.path import Path
            import matplotlib.patches as mpatches
            verts = [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor='none',
                                        edgecolor=COLORS['secondary'],
                                        lw=lw, alpha=alpha)
            ax.add_patch(patch)

    # Draw nodes
    for step, (x, y) in step_pos.items():
        freq = sum(t['count'] for t in transitions if t['from'] == step or t['to'] == step)
        size = 800 + (freq / max_count) * 2000
        ax.scatter(x, y, s=size, c=COLORS['primary'], zorder=5,
                   edgecolors='white', linewidths=2, alpha=0.85)
        # Label
        offset_x = -0.06 if x < 0.3 else (0.06 if x > 0.7 else 0)
        ha = 'right' if x < 0.3 else ('left' if x > 0.7 else 'center')
        va = 'center' if offset_x != 0 else 'bottom'
        ax.text(x + offset_x, y + (0.02 if va == 'bottom' else 0),
                step, fontsize=9, ha=ha, va=va, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Column headers
    col_labels = ['起始步骤', '中间步骤', '终止步骤']
    for ci, label in enumerate(col_labels):
        ax.text(col_x[ci], 0.98, label, ha='center', va='top',
                fontsize=13, fontweight='bold', color=COLORS['primary'])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axis('off')
    ax.set_title('AI教育技术路径流向图', fontsize=18, fontweight='bold', pad=20)
    fig.tight_layout()
    save_fig(fig, 'fig15_tech_pathway_sankey')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 16: Scenario x Stage Bubble Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig16_scenario_stage_bubble():
    print('Creating FIG 16: Scenario x Stage bubble...')
    data = load_json('cross_tabulations.json')
    ct = data['stage_scenario_l1']
    df = pd.DataFrame(ct['data'], index=ct['index'], columns=ct['columns'])

    main_stages = ['小学', '初中', '高中', '幼儿园', '中职']
    main_stages = [s for s in main_stages if s in df.index]
    scenarios = [c for c in df.columns if c != '未提及']
    df_plot = df.loc[main_stages, scenarios]

    fig, ax = plt.subplots(figsize=(12, 8))

    scenario_colors = {
        '助学': '#2E86C1', '助教': '#17A589', '助育': '#E67E22',
        '助评': '#C0392B', '助管': '#8E44AD', '助研': '#2C3E50'
    }

    max_val = df_plot.values.max()
    for i, stage in enumerate(main_stages):
        for j, scenario in enumerate(scenarios):
            val = df_plot.loc[stage, scenario]
            if val > 0:
                size = (val / max_val) * 2500 + 50
                color = scenario_colors.get(scenario, '#95A5A6')
                ax.scatter(j, i, s=size, c=color, alpha=0.7,
                          edgecolors='white', linewidths=1.5)
                if val > max_val * 0.05:
                    ax.text(j, i, str(int(val)), ha='center', va='center',
                            fontsize=9, fontweight='bold', color='white')

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_yticks(range(len(main_stages)))
    ax.set_yticklabels(main_stages, fontsize=12)
    ax.set_title('应用场景 × 学段 气泡图', fontsize=16, fontweight='bold', pad=20)

    # Size legend
    for val_label in [50, 200, 500]:
        ax.scatter([], [], s=(val_label / max_val) * 2500 + 50,
                   c='gray', alpha=0.5, label=f'{val_label}')
    ax.legend(title='案例数', loc='upper right', fontsize=9, title_fontsize=10)

    ax.set_xlim(-0.5, len(scenarios) - 0.5)
    ax.set_ylim(-0.5, len(main_stages) - 0.5)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save_fig(fig, 'fig16_scenario_stage_bubble')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 17: Innovation Cluster Scatter (PCA)
# ══════════════════════════════════════════════════════════════════════════════
def fig17_innovation_cluster():
    print('Creating FIG 17: Innovation clusters...')
    data = load_json('nlp_clusters.json')
    clusters = data['clusters']
    best_k = data['best_k']

    # We need to recreate PCA from the CSV data
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import jieba

    df = load_csv()
    # Use innovation text per case
    case_texts = df.groupby('案例编号').agg({
        '优势和创新点': lambda x: ' '.join(x.dropna().astype(str)),
        '学段': 'first',
    }).reset_index()

    texts = case_texts['优势和创新点'].tolist()
    stages = case_texts['学段'].tolist()

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())

    # KMeans
    km = KMeans(n_clusters=min(best_k, 8), random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)

    fig, ax = plt.subplots(figsize=(12, 10))

    cluster_colors = PALETTE_EXTENDED[:best_k]
    # Get cluster labels from data
    cluster_names = {}
    for cid, cdata in clusters.items():
        top_terms = cdata.get('top_terms', [])
        if top_terms:
            name = '/'.join([t['word'] for t in top_terms[:2]])
        else:
            name = f'Cluster {cid}'
        cluster_names[int(cid)] = name

    for ci in range(min(best_k, 8)):
        mask = labels == ci
        color = cluster_colors[ci % len(cluster_colors)]
        name = cluster_names.get(ci, f'Cluster {ci}')
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=30,
                   alpha=0.6, label=f'{name} (n={mask.sum()})',
                   edgecolors='white', linewidths=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('AI教育案例创新聚类分析（PCA降维）', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=8, ncol=2,
              bbox_to_anchor=(0, 1), framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save_fig(fig, 'fig17_innovation_cluster')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 18: Comprehensive Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def fig18_comprehensive_dashboard():
    print('Creating FIG 18: Comprehensive dashboard...')

    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Province Top 10 ──
    ax_a = fig.add_subplot(gs[0, 0:2])
    prov_data = load_json('province_distribution.json')['case_count_by_province']
    prov_data = {k: v for k, v in prov_data.items() if k != '未提及'}
    top10_prov = sorted(prov_data.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_prov.reverse()
    names = [x[0] for x in top10_prov]
    vals = [x[1] for x in top10_prov]
    norm = Normalize(vmin=min(vals), vmax=max(vals))
    colors = [GRADIENT_BLUES(norm(v)) for v in vals]
    ax_a.barh(range(len(names)), vals, color=colors, edgecolor='white')
    ax_a.set_yticks(range(len(names)))
    ax_a.set_yticklabels(names, fontsize=9)
    for i, v in enumerate(vals):
        ax_a.text(v + 5, i, str(v), va='center', fontsize=8)
    ax_a.set_title('A. 省份分布 Top 10', fontsize=12, fontweight='bold')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # ── Panel B: School Stage Donut ──
    ax_b = fig.add_subplot(gs[0, 2])
    stage_data = load_json('stage_distribution.json')['case_count_by_stage']
    main_s = {'小学': stage_data.get('小学', 0), '初中': stage_data.get('初中', 0),
              '高中': stage_data.get('高中', 0), '幼儿园': stage_data.get('幼儿园', 0)}
    other_s = sum(v for k, v in stage_data.items() if k not in main_s)
    main_s['其他'] = other_s
    s_labels = list(main_s.keys())
    s_vals = list(main_s.values())
    s_colors = PALETTE_MAIN[:len(s_labels)]
    wedges, _, autotexts = ax_b.pie(s_vals, labels=s_labels, autopct='%1.0f%%',
                                     colors=s_colors, startangle=90,
                                     wedgeprops=dict(width=0.4, edgecolor='white'),
                                     textprops={'fontsize': 8})
    for at in autotexts:
        at.set_fontsize(7)
    ax_b.set_title('B. 学段分布', fontsize=12, fontweight='bold')

    # ── Panel C: Scenario L1 ──
    ax_c = fig.add_subplot(gs[0, 3])
    scen = load_json('scenario_analysis.json')['scenario_level1']
    scen = {k: v for k, v in scen.items() if k != '未提及'}
    sc_labels = list(scen.keys())
    sc_vals = list(scen.values())
    sc_colors = ['#2E86C1', '#17A589', '#E67E22', '#C0392B', '#8E44AD', '#2C3E50'][:len(sc_labels)]
    wedges_c, _, autotexts_c = ax_c.pie(sc_vals, labels=sc_labels, autopct='%1.0f%%',
                                         colors=sc_colors, startangle=90,
                                         wedgeprops=dict(width=0.4, edgecolor='white'),
                                         textprops={'fontsize': 8})
    for at in autotexts_c:
        at.set_fontsize(7)
    ax_c.set_title('C. 应用场景（一级）', fontsize=12, fontweight='bold')

    # ── Panel D: Top 10 Tools ──
    ax_d = fig.add_subplot(gs[1, 0:2])
    tools = load_json('tool_product_distribution.json')['top30_tools']
    top10_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_tools.reverse()
    t_names = [x[0] for x in top10_tools]
    t_vals = [x[1] for x in top10_tools]
    t_norm = Normalize(vmin=min(t_vals), vmax=max(t_vals))
    t_colors = [GRADIENT_BLUES(t_norm(v)) for v in t_vals]
    ax_d.barh(range(len(t_names)), t_vals, color=t_colors, edgecolor='white')
    ax_d.set_yticks(range(len(t_names)))
    ax_d.set_yticklabels(t_names, fontsize=9)
    for i, v in enumerate(t_vals):
        ax_d.text(v + 3, i, str(v), va='center', fontsize=8)
    ax_d.set_title('D. Top 10 AI工具', fontsize=12, fontweight='bold')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # ── Panel E: Cultivation Radar ──
    ax_e = fig.add_subplot(gs[1, 2:4], polar=True)
    cult = load_json('nlp_cultivation.json')['overall']
    wuyu = ['智育', '德育', '美育', '体育', '劳育']
    wu_vals = [cult.get(w, 0) for w in wuyu]
    total_wu = sum(wu_vals) if sum(wu_vals) > 0 else 1
    wu_pct = [v / total_wu * 100 for v in wu_vals]
    angles_e = np.linspace(0, 2 * np.pi, len(wuyu), endpoint=False).tolist()
    wu_pct_plot = wu_pct + wu_pct[:1]
    angles_e_plot = angles_e + angles_e[:1]
    ax_e.plot(angles_e_plot, wu_pct_plot, 'o-', linewidth=2, color=COLORS['primary'], markersize=8)
    ax_e.fill(angles_e_plot, wu_pct_plot, alpha=0.2, color=COLORS['primary'])
    ax_e.set_xticks(angles_e)
    ax_e.set_xticklabels(wuyu, fontsize=10, fontweight='bold')
    ax_e.set_title('E. 五育培养方向', fontsize=12, fontweight='bold', pad=20)

    # PLACEHOLDER_DASHBOARD_BOTTOM
    # ── Panel F: Self-developed ratio ──
    ax_f = fig.add_subplot(gs[2, 0])
    sr = load_json('self_developed_ratio.json')
    sr_dist = sr['distribution']
    sr_labels = ['第三方', '自研']
    sr_vals = [sr_dist.get('FALSE', 0), sr_dist.get('TRUE', 0)]
    sr_colors = [COLORS['secondary'], COLORS['accent1']]
    ax_f.pie(sr_vals, labels=sr_labels, autopct='%1.1f%%', colors=sr_colors,
             startangle=90, wedgeprops=dict(width=0.45, edgecolor='white'),
             textprops={'fontsize': 9})
    ax_f.set_title('F. 自研比例', fontsize=12, fontweight='bold')

    # ── Panel G: Stage x Subject mini heatmap ──
    ax_g = fig.add_subplot(gs[2, 1:3])
    ct = load_json('cross_tabulations.json')['stage_subject']
    df_ct = pd.DataFrame(ct['data'], index=ct['index'], columns=ct['columns'])
    m_stages = [s for s in ['小学', '初中', '高中', '幼儿园'] if s in df_ct.index]
    top_subj = df_ct.loc[m_stages].sum().sort_values(ascending=False).head(8).index.tolist()
    top_subj = [s for s in top_subj if s != '未提及'][:7]
    df_mini = df_ct.loc[m_stages, top_subj]
    im = ax_g.imshow(df_mini.values, cmap=GRADIENT_BLUES, aspect='auto')
    ax_g.set_xticks(range(len(top_subj)))
    ax_g.set_xticklabels(top_subj, fontsize=9, rotation=30, ha='right')
    ax_g.set_yticks(range(len(m_stages)))
    ax_g.set_yticklabels(m_stages, fontsize=9)
    for ii in range(len(m_stages)):
        for jj in range(len(top_subj)):
            val = df_mini.values[ii, jj]
            c = 'white' if val > df_mini.values.max() * 0.6 else 'black'
            ax_g.text(jj, ii, str(int(val)), ha='center', va='center', fontsize=8, color=c)
    ax_g.set_title('G. 学段 x 学科', fontsize=12, fontweight='bold')

    # ── Panel H: Key stats ──
    ax_h = fig.add_subplot(gs[2, 3])
    ax_h.axis('off')
    stats_text = (
        f"数据概览\n"
        f"{'─'*20}\n"
        f"总记录数:  3,815\n"
        f"独立案例:  1,690\n"
        f"覆盖省份:  30\n"
        f"AI工具数:  1,830\n"
        f"自研比例:  14.4%\n"
        f"Top5集中度: 28.2%\n"
        f"{'─'*20}\n"
        f"主要学段: 小学(890)\n"
        f"主要场景: 助学(2804)\n"
        f"主要工具: 豆包(387)"
    )
    ax_h.text(0.1, 0.95, stats_text, transform=ax_h.transAxes,
              fontsize=11, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.3))
    ax_h.set_title('H. 关键指标', fontsize=12, fontweight='bold')

    fig.suptitle('AI教育产业研究 - 综合数据仪表盘',
                 fontsize=22, fontweight='bold', y=0.98)
    save_fig(fig, 'fig18_comprehensive_dashboard')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('='*60)
    print('AI Education Research - Visualization Suite')
    print(f'Output directory: {FIG_DIR}')
    print('='*60)

    funcs = [
        ('FIG 01', fig01_province_heatmap),
        ('FIG 02', fig02_school_stage_donut),
        ('FIG 03', fig03_top20_tools_bar),
        ('FIG 04', fig04_product_ecosystem_treemap),
        ('FIG 05', fig05_scenario_sunburst),
        ('FIG 06', fig06_subject_distribution),
        ('FIG 07', fig07_cultivation_radar),
        ('FIG 08', fig08_tech_wordcloud),
        ('FIG 09', fig09_tech_cooccurrence_network),
        ('FIG 10', fig10_stage_subject_heatmap),
        ('FIG 11', fig11_industry_maturity),
        ('FIG 12', fig12_self_developed_ratio),
        ('FIG 13', fig13_company_market_concentration),
        ('FIG 14', fig14_lda_topics),
        ('FIG 15', fig15_tech_pathway_sankey),
        ('FIG 16', fig16_scenario_stage_bubble),
        ('FIG 17', fig17_innovation_cluster),
        ('FIG 18', fig18_comprehensive_dashboard),
    ]

    success = 0
    failed = []
    for name, func in funcs:
        try:
            func()
            success += 1
        except Exception as e:
            print(f'  ERROR in {name}: {e}')
            import traceback
            traceback.print_exc()
            failed.append(name)

    print('\n' + '='*60)
    print(f'Completed: {success}/{len(funcs)} figures')
    if failed:
        print(f'Failed: {", ".join(failed)}')
    print(f'Output: {FIG_DIR}')
    print('='*60)


if __name__ == '__main__':
    main()
