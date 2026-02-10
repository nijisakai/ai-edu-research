#!/usr/bin/env python3
"""Framework-based visualization for AI Education Research.
Generates figures based on theoretical framework dimensions (三赋能, iSTAR, 数字教学法, etc.)
from the V6 CSV with framework annotations.
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
CSV_PATH = '/Users/sakai/Desktop/产业调研/ai-edu-research/output/教育产品统计_V6_框架标注.csv'
STATS_PATH = '/Users/sakai/Desktop/产业调研/ai-edu-research/output/framework_stats.json'
OUTPUT_DIR = Path('/Users/sakai/Desktop/产业调研/ai-edu-research/output')
FIG_DIR = OUTPUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.sans-serif': ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
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
PALETTE_6 = [PRIMARY, SECONDARY, ACCENT, WARM, TEAL, PURPLE]
PALETTE_8 = [PRIMARY, SECONDARY, TEAL, '#27AE60', WARM, '#E67E22', ACCENT, PURPLE]
GRADIENT_BLUES = ['#D6EAF8', '#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1', '#2874A6', '#21618C', '#1B4F72']

SOURCE_TEXT = '数据来源: 1,690个AI+基础教育案例 | 理论框架: 黄荣怀数字教学法/iSTAR/三赋能'

def _remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _add_caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha='center', va='top', fontsize=10,
             fontstyle='italic', color='#555555',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                       edgecolor='#DEE2E6', alpha=0.9))

def _add_source(fig, y=0.005):
    fig.text(0.98, y, SOURCE_TEXT, ha='right', va='bottom', fontsize=7,
             color='#999999', fontstyle='italic')

def _add_analysis(fig, text, x=0.02, y=0.02):
    fig.text(x, y, text, ha='left', va='bottom', fontsize=8,
             color='#1a5276', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                       edgecolor='#cccccc', alpha=0.8))

def _save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  [OK] {name}')

# ── Load Data ──────────────────────────────────────────────────────────
print('Loading V6 CSV and framework stats...')
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
with open(STATS_PATH, 'r', encoding='utf-8') as f:
    fstats = json.load(f)
print(f'  Loaded {len(df)} rows')

# ═══════════════════════════════════════════════════════════════════════
# FIG F1 — 三赋能×iSTAR交叉热力图
# ═══════════════════════════════════════════════════════════════════════
def fig_f1():
    print('Fig_F1: 三赋能×iSTAR heatmap...')
    cross = fstats['三赋能_x_iSTAR']

    istar_order = ['HUM(0)', 'HMC(1)', 'HM2C(2)']
    sf_order = ['赋能学生', '赋能教师', '赋能评价', '赋能学校']

    # Build matrix
    matrix = np.zeros((len(sf_order), len(istar_order)))
    for j, ist in enumerate(istar_order):
        for i, sf in enumerate(sf_order):
            matrix[i, j] = cross.get(ist, {}).get(sf, 0)

    # Percentage matrix (row-wise)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    pct_matrix = matrix / row_totals * 100

    fig, ax = plt.subplots(figsize=(14, 10))

    cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['#FDFEFE', '#AED6F1', '#2E86C1', '#1B4F72'])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    # Annotate cells
    for i in range(len(sf_order)):
        for j in range(len(istar_order)):
            val = int(matrix[i, j])
            pct = pct_matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.5 else '#333'
            ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(istar_order)))
    ax.set_xticklabels(['HUM(0)\n人类主导', 'HMC(1)\n人机协作', 'HM2C(2)\n人机共创'],
                       fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(sf_order)))
    ax.set_yticklabels(sf_order, fontsize=12, fontweight='bold')

    ax.set_xlabel('iSTAR人机协同层级', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('三赋能分类', fontsize=13, fontweight='bold', labelpad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('案例数量', fontsize=11)

    # Row totals on right
    for i, sf in enumerate(sf_order):
        total = int(matrix[i].sum())
        ax.text(len(istar_order) - 0.3, i, f'  合计: {total}', ha='left', va='center',
                fontsize=9, color='#666', fontstyle='italic')

    ax.set_title('三赋能×人机协同层级交叉分析', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    # Annotation box
    insight = ('关键发现:\n'
               '- 赋能学生在HM2C层级最集中(1571例)\n'
               '- 赋能教师主要在HMC层级(679例)\n'
               '- 赋能学校几乎全部在HUM层级(81例)\n'
               '- 人机共创主要服务于学生端')
    props = dict(boxstyle='round,pad=0.6', facecolor='#FEF9E7', edgecolor=WARM, alpha=0.95)
    ax.text(1.35, 0.95, insight, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='left', bbox=props, color='#333', linespacing=1.5)

    fig.subplots_adjust(top=0.90, bottom=0.12, right=0.78)
    _add_caption(fig, '图F1 三赋能×人机协同层级交叉分析', y=0.02)
    _add_source(fig)
    _save(fig, 'fig_f1_sanfuneng_istar_heatmap.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F2 — 技术代际演进河流图 (Stacked Area)
# ═══════════════════════════════════════════════════════════════════════
def fig_f2():
    print('Fig_F2: Tech generation streamgraph...')
    cross = fstats['三赋能_x_技术代际']

    gen_order = ['Gen1_传统信息化', 'Gen2_互联网+', 'Gen3_AI辅助', 'Gen4_大模型', 'Gen5_多模态AI']
    gen_short = ['Gen1\n传统信息化', 'Gen2\n互联网+', 'Gen3\nAI辅助', 'Gen4\n大模型', 'Gen5\n多模态AI']
    sf_order = ['赋能学生', '赋能教师', '赋能评价', '赋能学校']
    sf_colors = {'赋能学生': '#3498DB', '赋能教师': '#E74C3C', '赋能评价': '#F39C12', '赋能学校': '#27AE60'}

    # Build data arrays
    data = {sf: [] for sf in sf_order}
    for gen in gen_order:
        gen_data = cross.get(gen, {})
        for sf in sf_order:
            data[sf].append(gen_data.get(sf, 0))

    x = np.arange(len(gen_order))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Stacked area chart
    bottom = np.zeros(len(gen_order))
    for sf in sf_order:
        vals = np.array(data[sf], dtype=float)
        ax.fill_between(x, bottom, bottom + vals, alpha=0.7, color=sf_colors[sf], label=sf)
        ax.plot(x, bottom + vals, color=sf_colors[sf], linewidth=1.5, alpha=0.9)
        # Label in the middle of each area for the largest generation
        mid = bottom + vals / 2
        peak_idx = np.argmax(vals)
        if vals[peak_idx] > 50:
            ax.text(peak_idx, mid[peak_idx], f'{sf}\n{int(vals[peak_idx])}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white', alpha=0.9)
        bottom += vals

    # Total line on top
    totals = bottom
    ax.plot(x, totals, 'k--', linewidth=1.5, alpha=0.4)
    for i, t in enumerate(totals):
        ax.text(i, t + 15, f'{int(t)}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(gen_short, fontsize=10, fontweight='bold')
    ax.set_ylabel('案例数量', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, len(gen_order) - 0.7)
    _remove_spines(ax)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    ax.set_title('技术代际演进与赋能方向分布', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    # Trend arrow
    ax.annotate('技术演进方向', xy=(4.2, totals[-1] * 0.5), xytext=(0.5, totals[-1] * 0.5),
                fontsize=10, color='#999', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.5))

    fig.subplots_adjust(top=0.92, bottom=0.10)
    _add_caption(fig, '图F2 技术代际演进与赋能方向分布', y=0.01)
    _add_source(fig)
    _add_analysis(fig, 'Gen4大模型和Gen2互联网+是两大主力代际，Gen5多模态AI快速崛起，赋能学生始终是各代际核心方向。', x=0.05, y=0.08)
    _save(fig, 'fig_f2_tech_generation_stream.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F3 — 数字教学法四维雷达对比图
# ═══════════════════════════════════════════════════════════════════════
def fig_f3():
    print('Fig_F3: Digital pedagogy radar...')
    dims = ['D1_深度学习', 'D2_绿色鲁棒', 'D3_循证教学', 'D4_人机互信']
    dim_labels = ['D1 深度学习', 'D2 绿色鲁棒', 'D3 循证教学', 'D4 人机互信']
    stages = ['小学', '初中', '高中', '幼儿园']
    stage_colors = {'小学': PRIMARY, '初中': SECONDARY, '高中': ACCENT, '幼儿园': WARM}

    # Compute per-stage dimension counts
    stage_data = {s: {d: 0 for d in dims} for s in stages}
    stage_totals = {s: 0 for s in stages}
    for _, row in df.iterrows():
        st = str(row.get('学段', ''))
        if st in stages:
            stage_totals[st] += 1
            for d in dims:
                val = row.get(d, 0)
                try:
                    if int(val) == 1:
                        stage_data[st][d] += 1
                except (ValueError, TypeError):
                    pass

    # Normalize to proportions
    for s in stages:
        total = stage_totals[s] if stage_totals[s] > 0 else 1
        for d in dims:
            stage_data[s][d] = stage_data[s][d] / total

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))

    for st in stages:
        values = [stage_data[st][d] for d in dims]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, alpha=0.85,
                color=stage_colors[st], label=st, markersize=7)
        ax.fill(angles, values, alpha=0.1, color=stage_colors[st])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'], fontsize=8, color='#999')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, framealpha=0.9)
    ax.set_title('不同学段数字教学法四维分布', fontsize=16, fontweight='bold', color=PRIMARY, pad=30)
    ax.grid(True, alpha=0.25)

    # Annotation
    insight = ('维度解读:\n'
               'D1深度学习: 促进高阶思维\n'
               'D2绿色鲁棒: 安全可靠使用\n'
               'D3循证教学: 数据驱动决策\n'
               'D4人机互信: 师生信任AI')
    props = dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', edgecolor=SECONDARY, alpha=0.9)
    fig.text(0.88, 0.15, insight, fontsize=8, va='bottom', ha='left',
             bbox=props, color='#333', linespacing=1.5)

    fig.subplots_adjust(bottom=0.08)
    _add_caption(fig, '图F3 不同学段数字教学法四维分布', y=0.01)
    _add_source(fig)
    _add_analysis(fig, 'D1深度学习在各学段占比最高，D2绿色鲁棒普遍薄弱，反映AI安全意识尚需加强。', x=0.05, y=0.08)
    _save(fig, 'fig_f3_digital_pedagogy_radar.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F4 — 智慧教育三重境界×学段桑基图
# ═══════════════════════════════════════════════════════════════════════
def fig_f4():
    print('Fig_F4: Smart education sankey...')
    stages = ['幼儿园', '小学', '初中', '高中']
    realms = ['第一境界_智慧环境', '第二境界_教学模式', '第三境界_制度变革']
    paths = ['T1_内容生成', 'T2_智能评测', 'T3_数据驱动', 'T4_沉浸体验', 'T5_智能硬件', 'T6_平台生态']

    realm_short = {r: r.split('_')[1] for r in realms}
    path_short = {p: p.split('_')[1] for p in paths}

    # Cross-tabulate: 学段→境界, 境界→技术路径
    cross_sr = pd.crosstab(df['学段'], df['智慧教育境界'])
    cross_rp = pd.crosstab(df['智慧教育境界'], df['技术路径类型'])

    fig, ax = plt.subplots(figsize=(16, 12))

    # Three columns: 学段 (x=0.08), 境界 (x=0.45), 技术路径 (x=0.82)
    col_x = [0.08, 0.45, 0.82]

    # Position nodes
    stage_y = np.linspace(0.85, 0.15, len(stages))
    realm_y = np.linspace(0.85, 0.15, len(realms))
    path_y = np.linspace(0.90, 0.10, len(paths))

    stage_pos = {s: (col_x[0], stage_y[i]) for i, s in enumerate(stages)}
    realm_pos = {r: (col_x[1], realm_y[i]) for i, r in enumerate(realms)}
    path_pos = {p: (col_x[2], path_y[i]) for i, p in enumerate(paths)}

    # Color maps
    stage_colors = {'幼儿园': WARM, '小学': PRIMARY, '初中': SECONDARY, '高中': ACCENT}
    realm_colors = {'第一境界_智慧环境': '#27AE60', '第二境界_教学模式': '#2E86C1', '第三境界_制度变革': '#8E44AD'}

    # Draw flows: 学段→境界
    max_sr = cross_sr.values.max() if cross_sr.size > 0 else 1
    for s in stages:
        for r in realms:
            v = cross_sr.loc[s, r] if s in cross_sr.index and r in cross_sr.columns else 0
            if v > 5:
                x0, y0 = stage_pos[s]
                x1, y1 = realm_pos[r]
                alpha = 0.1 + 0.4 * v / max_sr
                lw = 1 + 8 * v / max_sr
                xs = np.linspace(x0, x1, 50)
                t = (xs - x0) / (x1 - x0)
                ys = y0 + (y1 - y0) * (3 * t**2 - 2 * t**3)
                ax.plot(xs, ys, color=stage_colors[s], alpha=alpha, linewidth=lw, solid_capstyle='round')

    # Draw flows: 境界→技术路径
    max_rp = cross_rp.values.max() if cross_rp.size > 0 else 1
    for r in realms:
        for p in paths:
            v = cross_rp.loc[r, p] if r in cross_rp.index and p in cross_rp.columns else 0
            if v > 5:
                x0, y0 = realm_pos[r]
                x1, y1 = path_pos[p]
                alpha = 0.1 + 0.4 * v / max_rp
                lw = 1 + 8 * v / max_rp
                xs = np.linspace(x0, x1, 50)
                t = (xs - x0) / (x1 - x0)
                ys = y0 + (y1 - y0) * (3 * t**2 - 2 * t**3)
                ax.plot(xs, ys, color=realm_colors[r], alpha=alpha, linewidth=lw, solid_capstyle='round')

    # Draw nodes
    for s, (x, y) in stage_pos.items():
        total = cross_sr.loc[s].sum() if s in cross_sr.index else 0
        ax.scatter(x, y, s=300, color=stage_colors[s], zorder=5, edgecolors='white', linewidths=2)
        ax.text(x - 0.02, y, f'{s} ({int(total)})', ha='right', va='center',
                fontsize=10, fontweight='bold', color='#333')

    for r, (x, y) in realm_pos.items():
        total = cross_sr[r].sum() if r in cross_sr.columns else 0
        ax.scatter(x, y, s=350, color=realm_colors[r], zorder=5, edgecolors='white', linewidths=2)
        ax.text(x, y + 0.04, f'{realm_short[r]}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=realm_colors[r])
        ax.text(x, y - 0.04, f'({int(total)})', ha='center', va='top',
                fontsize=9, color='#666')

    for p, (x, y) in path_pos.items():
        total = cross_rp[p].sum() if p in cross_rp.columns else 0
        ax.scatter(x, y, s=250, color=TEAL, zorder=5, edgecolors='white', linewidths=2)
        ax.text(x + 0.02, y, f'{path_short[p]} ({int(total)})', ha='left', va='center',
                fontsize=10, fontweight='bold', color='#333')

    # Column headers
    ax.text(col_x[0], 0.95, '学段', ha='center', fontsize=14, fontweight='bold', color=PRIMARY)
    ax.text(col_x[1], 0.95, '智慧教育境界', ha='center', fontsize=14, fontweight='bold', color='#2E86C1')
    ax.text(col_x[2], 0.95, '技术路径', ha='center', fontsize=14, fontweight='bold', color=TEAL)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.02, 1.0)
    ax.axis('off')
    ax.set_title('学段→智慧教育境界→技术路径流向', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    fig.subplots_adjust(top=0.92, bottom=0.08)
    _add_caption(fig, '图F4 学段→智慧教育境界→技术路径流向', y=0.01)
    _add_source(fig)
    _add_analysis(fig, '第二境界(教学模式)是各学段主流，小学案例最多且主要流向T1内容生成和T6平台生态路径。', x=0.05, y=0.06)
    _save(fig, 'fig_f4_smart_edu_sankey.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F5 — 创新深度评分分布与影响因素
# ═══════════════════════════════════════════════════════════════════════
def fig_f5():
    print('Fig_F5: Innovation depth score...')
    stages = ['幼儿园', '小学', '初中', '高中']
    stage_colors = {'幼儿园': WARM, '小学': PRIMARY, '初中': SECONDARY, '高中': ACCENT}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # Left: Violin plot of 创新深度评分 by 学段
    score_col = '创新深度评分'
    df_valid = df[df[score_col].notna() & df['学段'].isin(stages)].copy()
    df_valid[score_col] = pd.to_numeric(df_valid[score_col], errors='coerce')
    df_valid = df_valid.dropna(subset=[score_col])

    positions = []
    data_arrays = []
    colors_list = []
    for i, st in enumerate(stages):
        subset = df_valid[df_valid['学段'] == st][score_col].values
        if len(subset) > 0:
            data_arrays.append(subset)
            positions.append(i)
            colors_list.append(stage_colors[st])

    if data_arrays:
        parts = ax1.violinplot(data_arrays, positions=positions, showmeans=True,
                               showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('#333')
        parts['cmedians'].set_color(ACCENT)

        # Add jittered points
        for i, (arr, pos) in enumerate(zip(data_arrays, positions)):
            jitter = np.random.normal(0, 0.05, len(arr))
            ax1.scatter(pos + jitter, arr, s=3, alpha=0.15, color=colors_list[i])

        # Mean labels
        for i, (arr, pos) in enumerate(zip(data_arrays, positions)):
            mean_val = np.mean(arr)
            ax1.text(pos, 5.3, f'M={mean_val:.2f}\nn={len(arr)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors_list[i])

    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax1.set_ylabel('创新深度评分', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.5, 5.8)
    ax1.set_title('各学段创新深度评分分布', fontsize=13, fontweight='bold', color=PRIMARY)
    _remove_spines(ax1)

    # Right: Horizontal bar showing mean score by 省份 (top 15)
    prov_scores = df_valid.groupby('省份')[score_col].agg(['mean', 'count'])
    prov_scores = prov_scores[prov_scores['count'] >= 10]  # Min 10 cases
    prov_scores = prov_scores.sort_values('mean', ascending=True).tail(15)

    norm = plt.Normalize(prov_scores['mean'].min(), prov_scores['mean'].max())
    cmap = mcolors.LinearSegmentedColormap.from_list('bg', ['#D6EAF8', PRIMARY])
    bar_colors = [cmap(norm(v)) for v in prov_scores['mean']]

    bars = ax2.barh(range(len(prov_scores)), prov_scores['mean'], color=bar_colors,
                    edgecolor='white', linewidth=0.5, height=0.7)
    ax2.set_yticks(range(len(prov_scores)))
    ax2.set_yticklabels(prov_scores.index, fontsize=9)
    ax2.set_xlabel('平均创新深度评分', fontsize=11, fontweight='bold')
    ax2.set_title('省份平均创新深度 (n>=10)', fontsize=13, fontweight='bold', color=PRIMARY)
    _remove_spines(ax2)

    # Value labels
    for i, (v, n) in enumerate(zip(prov_scores['mean'], prov_scores['count'])):
        ax2.text(v + 0.02, i, f'{v:.2f} (n={int(n)})', va='center', fontsize=8, color='#333')

    fig.suptitle('创新深度评分: 学段差异与地域分布', fontsize=16, fontweight='bold', color=PRIMARY, y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.35)
    _add_caption(fig, '图F5 创新深度评分: 学段差异与地域分布', y=0.01)
    _add_source(fig)
    _add_analysis(fig, f'全样本平均创新深度{fstats["创新深度评分_mean"]:.2f}分(满分5)，高中略高于小学，地域差异显著。', x=0.05, y=0.08)
    _save(fig, 'fig_f5_innovation_depth.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F6 — iSTAR层级×技术路径气泡矩阵
# ═══════════════════════════════════════════════════════════════════════
def fig_f6():
    print('Fig_F6: iSTAR × tech path bubble matrix...')
    cross = fstats['iSTAR_x_技术路径']

    path_order = ['T1_内容生成', 'T2_智能评测', 'T3_数据驱动', 'T4_沉浸体验', 'T5_智能硬件', 'T6_平台生态']
    istar_order = ['HUM(0)', 'HMC(1)', 'HM2C(2)']
    path_short = [p.split('_')[1] for p in path_order]
    istar_labels = ['HUM(0)\n人类主导', 'HMC(1)\n人机协作', 'HM2C(2)\n人机共创']

    # Build count matrix and mean innovation score matrix
    count_matrix = np.zeros((len(istar_order), len(path_order)))
    score_matrix = np.zeros((len(istar_order), len(path_order)))

    for j, p in enumerate(path_order):
        path_data = cross.get(p, {})
        for i, ist in enumerate(istar_order):
            count_matrix[i, j] = path_data.get(ist, 0)

    # Compute mean innovation score per cell from raw data
    for _, row in df.iterrows():
        ist = str(row.get('iSTAR人机协同层级', ''))
        p = str(row.get('技术路径类型', ''))
        score = row.get('创新深度评分', np.nan)
        if ist in istar_order and p in path_order:
            i = istar_order.index(ist)
            j = path_order.index(p)
            try:
                s = float(score)
                if not np.isnan(s):
                    # Accumulate for averaging
                    score_matrix[i, j] += s
            except (ValueError, TypeError):
                pass

    # Average scores
    for i in range(len(istar_order)):
        for j in range(len(path_order)):
            if count_matrix[i, j] > 0:
                score_matrix[i, j] /= count_matrix[i, j]

    fig, ax = plt.subplots(figsize=(14, 10))

    max_count = count_matrix.max()
    cmap = plt.cm.get_cmap('YlOrRd')
    score_min = score_matrix[score_matrix > 0].min() if (score_matrix > 0).any() else 1
    score_max = score_matrix.max()

    for i in range(len(istar_order)):
        for j in range(len(path_order)):
            v = count_matrix[i, j]
            s = score_matrix[i, j]
            if v > 0:
                size = 200 + 2500 * (v / max_count)
                color_val = (s - score_min) / (score_max - score_min) if score_max > score_min else 0.5
                color = cmap(0.2 + 0.7 * color_val)
                ax.scatter(j, i, s=size, c=[color], alpha=0.8,
                           edgecolors='#333', linewidths=1)
                # Count label
                ax.text(j, i - 0.02, f'{int(v)}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#333')
                # Score label below
                ax.text(j, i + 0.22, f'M={s:.2f}', ha='center', va='center',
                        fontsize=7.5, color='#666')

    ax.set_xticks(range(len(path_order)))
    ax.set_xticklabels(path_short, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(istar_order)))
    ax.set_yticklabels(istar_labels, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, len(path_order) - 0.5)
    ax.set_ylim(-0.5, len(istar_order) - 0.5)

    ax.set_xlabel('技术路径类型', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('iSTAR人机协同层级', fontsize=13, fontweight='bold', labelpad=10)
    _remove_spines(ax)

    # Size legend
    for sv, sl in [(100, '50'), (500, '200'), (1500, '800')]:
        ax.scatter([], [], s=sv, c='#ccc', edgecolors='#333', linewidths=1, label=f'{sl}例')
    ax.legend(title='案例数量', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.9)

    # Colorbar for score
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(score_min, score_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('平均创新深度评分', fontsize=10)

    ax.set_title('人机协同层级×技术路径矩阵', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    fig.subplots_adjust(top=0.92, bottom=0.10)
    _add_caption(fig, '图F6 人机协同层级×技术路径矩阵 (气泡大小=案例数, 颜色=创新深度)', y=0.01)
    _add_source(fig)
    _add_analysis(fig, 'HM2C×T1内容生成是最大热点(1239例)，HMC×T6平台生态次之(994例)，数据驱动路径案例最少。', x=0.05, y=0.08)
    _save(fig, 'fig_f6_istar_techpath_bubble.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F7 — 理论框架综合仪表盘
# ═══════════════════════════════════════════════════════════════════════
def fig_f7():
    print('Fig_F7: Framework dashboard...')
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('AI赋能基础教育理论框架综合视图', fontsize=18, fontweight='bold',
                 color=PRIMARY, y=0.98)

    # 2×3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35, top=0.92, bottom=0.08,
                          left=0.06, right=0.96)

    # ── Panel 1: 三赋能 donut ──
    ax1 = fig.add_subplot(gs[0, 0])
    sf_data = fstats['三赋能分类']
    sf_names = list(sf_data.keys())
    sf_vals = list(sf_data.values())
    sf_colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']
    wedges, texts = ax1.pie(sf_vals, colors=sf_colors[:len(sf_names)],
                            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                            startangle=90)
    # Center text
    total_sf = sum(sf_vals)
    ax1.text(0, 0, f'{total_sf}\n案例', ha='center', va='center', fontsize=11, fontweight='bold')
    # Legend below
    legend_text = '\n'.join([f'{n}: {v} ({v/total_sf*100:.1f}%)' for n, v in zip(sf_names, sf_vals)])
    ax1.text(0, -1.4, legend_text, ha='center', va='top', fontsize=7.5, color='#555')
    ax1.set_title('三赋能分类', fontsize=12, fontweight='bold', color=PRIMARY)

    # ── Panel 2: iSTAR层级 bar ──
    ax2 = fig.add_subplot(gs[0, 1])
    ist_data = fstats['iSTAR人机协同层级']
    ist_order = ['HUM(0)', 'HMC(1)', 'HM2C(2)']
    ist_vals = [ist_data.get(k, 0) for k in ist_order]
    ist_colors = ['#AED6F1', '#3498DB', '#1B4F72']
    bars = ax2.bar(range(3), ist_vals, color=ist_colors, edgecolor='white', width=0.6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['HUM', 'HMC', 'HM2C'], fontsize=9, fontweight='bold')
    for bar, v in zip(bars, ist_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(v), ha='center', fontsize=8, fontweight='bold')
    _remove_spines(ax2)
    ax2.set_title('iSTAR层级', fontsize=12, fontweight='bold', color=PRIMARY)

    # ── Panel 3: 技术代际 stacked bar ──
    ax3 = fig.add_subplot(gs[0, 2])
    gen_data = fstats['产品技术代际']
    gen_order = ['Gen1_传统信息化', 'Gen2_互联网+', 'Gen3_AI辅助', 'Gen4_大模型', 'Gen5_多模态AI']
    gen_vals = [gen_data.get(k, 0) for k in gen_order]
    gen_short = ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5']
    gen_colors = ['#D5F5E3', '#82E0AA', '#2ECC71', '#27AE60', '#145A32']
    bars = ax3.bar(range(5), gen_vals, color=gen_colors, edgecolor='white', width=0.6)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(gen_short, fontsize=8, fontweight='bold')
    for bar, v in zip(bars, gen_vals):
        if v > 50:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    str(v), ha='center', fontsize=7, fontweight='bold')
    _remove_spines(ax3)
    ax3.set_title('技术代际', fontsize=12, fontweight='bold', color=PRIMARY)

    # ── Panel 4: 智慧教育境界 pie ──
    ax4 = fig.add_subplot(gs[1, 0])
    realm_data = fstats['智慧教育境界']
    realm_names = list(realm_data.keys())
    realm_vals = list(realm_data.values())
    realm_short = [n.split('_')[1] for n in realm_names]
    realm_colors = ['#27AE60', '#2E86C1', '#8E44AD']
    wedges4, texts4, autotexts4 = ax4.pie(realm_vals, labels=realm_short,
                                           colors=realm_colors[:len(realm_names)],
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 8})
    for at in autotexts4:
        at.set_fontsize(8)
        at.set_fontweight('bold')
    ax4.set_title('智慧教育境界', fontsize=12, fontweight='bold', color=PRIMARY)

    # ── Panel 5: 创新深度 histogram ──
    ax5 = fig.add_subplot(gs[1, 1])
    score_data = fstats['创新深度评分']
    scores = list(range(1, 6))
    score_vals = [score_data.get(str(s), 0) for s in scores]
    score_colors = ['#FADBD8', '#F1948A', '#E74C3C', '#CB4335', '#922B21']
    bars5 = ax5.bar(scores, score_vals, color=score_colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars5, score_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(v), ha='center', fontsize=8, fontweight='bold')
    ax5.set_xlabel('评分', fontsize=9)
    ax5.set_xticks(scores)
    _remove_spines(ax5)
    ax5.set_title(f'创新深度 (M={fstats["创新深度评分_mean"]:.2f})', fontsize=12,
                  fontweight='bold', color=PRIMARY)

    # ── Panel 6: 数字教学法 radar ──
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    dim_data = fstats['数字教学法维度']
    dim_names = ['D1_深度学习', 'D2_绿色鲁棒', 'D3_循证教学', 'D4_人机互信']
    dim_labels = ['D1深度学习', 'D2绿色鲁棒', 'D3循证教学', 'D4人机互信']
    dim_vals = [dim_data.get(d, 0) for d in dim_names]
    total_dim = sum(dim_vals)
    dim_pct = [v / total_dim for v in dim_vals]

    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
    angles += angles[:1]
    dim_pct_plot = dim_pct + dim_pct[:1]

    ax6.plot(angles, dim_pct_plot, 'o-', linewidth=2, color=SECONDARY, markersize=6)
    ax6.fill(angles, dim_pct_plot, alpha=0.2, color=SECONDARY)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(dim_labels, fontsize=8, fontweight='bold')
    ax6.set_ylim(0, max(dim_pct) * 1.2)
    ax6.set_yticklabels([])
    ax6.set_title('数字教学法', fontsize=12, fontweight='bold', color=PRIMARY, pad=15)

    _add_caption(fig, '图F7 AI赋能基础教育理论框架综合视图', y=0.02)
    _add_source(fig, y=0.01)
    _save(fig, 'fig_f7_framework_dashboard.png')

# ═══════════════════════════════════════════════════════════════════════
# FIG F8 — 省份×理论框架多维对比
# ═══════════════════════════════════════════════════════════════════════
def fig_f8():
    print('Fig_F8: Province framework comparison...')
    # Get top 10 provinces by case count
    prov_counts = df['省份'].value_counts()
    top_provs = prov_counts.head(10).index.tolist()

    # Compute metrics per province
    metrics = {}
    for prov in top_provs:
        sub = df[df['省份'] == prov]
        n = len(sub)

        # Mean iSTAR level (HUM=0, HMC=1, HM2C=2)
        istar_map = {'HUM(0)': 0, 'HMC(1)': 1, 'HM2C(2)': 2}
        istar_vals = sub['iSTAR人机协同层级'].map(istar_map).dropna()
        mean_istar = istar_vals.mean() if len(istar_vals) > 0 else 0

        # Mean innovation depth
        scores = pd.to_numeric(sub['创新深度评分'], errors='coerce').dropna()
        mean_score = scores.mean() if len(scores) > 0 else 0

        # % Gen4+Gen5
        gen45 = sub['产品技术代际'].isin(['Gen4_大模型', 'Gen5_多模态AI']).sum()
        pct_gen45 = gen45 / n * 100 if n > 0 else 0

        # % HM2C
        hm2c = (sub['iSTAR人机协同层级'] == 'HM2C(2)').sum()
        pct_hm2c = hm2c / n * 100 if n > 0 else 0

        metrics[prov] = {
            'mean_istar': mean_istar,
            'mean_score': mean_score,
            'pct_gen45': pct_gen45,
            'pct_hm2c': pct_hm2c,
            'n': n
        }

    fig, ax = plt.subplots(figsize=(16, 10))

    metric_names = ['平均iSTAR层级', '平均创新深度', 'Gen4+5占比(%)', 'HM2C占比(%)']
    metric_keys = ['mean_istar', 'mean_score', 'pct_gen45', 'pct_hm2c']
    bar_colors = [PRIMARY, ACCENT, TEAL, PURPLE]

    # Normalize each metric to 0-1 for comparison
    raw_vals = {k: [metrics[p][k] for p in top_provs] for k in metric_keys}
    norm_vals = {}
    for k in metric_keys:
        vals = raw_vals[k]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax > vmin else 1
        norm_vals[k] = [(v - vmin) / rng for v in vals]

    x = np.arange(len(top_provs))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (mname, mkey) in enumerate(zip(metric_names, metric_keys)):
        vals = norm_vals[mkey]
        bars = ax.bar(x + offsets[i] * width, vals, width, color=bar_colors[i],
                      alpha=0.85, label=mname, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}\n(n={metrics[p]["n"]})' for p in top_provs],
                       fontsize=9, fontweight='bold')
    ax.set_ylabel('标准化指标值 (0-1)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    _remove_spines(ax)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

    ax.set_title('重点省份理论框架指标对比', fontsize=16, fontweight='bold', color=PRIMARY, pad=20)

    # Add raw value table below
    table_text = '原始值: '
    for p in top_provs[:5]:
        m = metrics[p]
        table_text += f'{p}(iSTAR={m["mean_istar"]:.1f}, 创新={m["mean_score"]:.1f}, Gen4+5={m["pct_gen45"]:.0f}%) | '
    ax.text(0.5, -0.12, table_text, transform=ax.transAxes, ha='center',
            fontsize=7, color='#666', fontstyle='italic')

    fig.subplots_adjust(top=0.92, bottom=0.14)
    _add_caption(fig, '图F8 重点省份理论框架指标对比 (标准化)', y=0.01)
    _add_source(fig)
    _add_analysis(fig, '各省份在四个框架维度上表现差异显著，发达地区在技术代际和人机协同层级上领先。', x=0.05, y=0.08)
    _save(fig, 'fig_f8_province_framework.png')


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('Framework Visualization (V6)')
    print('='*60)

    funcs = [fig_f1, fig_f2, fig_f3, fig_f4, fig_f5, fig_f6, fig_f7, fig_f8]

    for fn in funcs:
        try:
            fn()
        except Exception as e:
            print(f'  [ERROR] {fn.__name__}: {e}')
            import traceback
            traceback.print_exc()

    print('='*60)
    print(f'Done. Framework figures saved to {FIG_DIR}')
    print('='*60)
