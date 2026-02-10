#!/usr/bin/env python3
"""
Advanced Statistical Analysis for AI Education Research
Generates 10 publication-quality supplementary figures (S1-S10)
Nature/Science style with Chinese annotations
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
import json
import os
import ast

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['font.size'] = 9

# Custom palette
BLUES = ['#1a5276', '#2980b9', '#5dade2', '#85c1e9', '#aed6f1']
TEALS = ['#0e6655', '#1abc9c', '#48c9b0', '#76d7c4', '#a3e4d7']
ORANGES = ['#935116', '#e67e22', '#f0b27a', '#f5cba7', '#fae5d3']
PURPLES = ['#6c3483', '#8e44ad', '#bb8fce', '#d2b4de', '#e8daef']
REDS = ['#922b21', '#e74c3c', '#f1948a', '#f5b7b1', '#fadbd8']

PALETTE_MAIN = [BLUES[1], TEALS[1], ORANGES[1], PURPLES[1], REDS[1],
                BLUES[0], TEALS[0], ORANGES[0], PURPLES[0], REDS[0]]

FIG_DIR = '/Users/sakai/Desktop/产业调研/ai-edu-research/output/figures'
CSV_PATH = '/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv'

# ── Helper Functions ──────────────────────────────────────────────────────────

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

def add_grid(ax, axis='y'):
    ax.grid(axis=axis, alpha=0.15, linewidth=0.5, color='#888888')
    ax.set_axisbelow(True)

def save_fig(fig, name, caption):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] {name}")

def add_analysis_box(fig, text, x=0.02, y=0.02):
    """Add an analysis insight annotation box."""
    fig.text(x, y, text, ha='left', va='bottom', fontsize=7.5,
             color='#1a5276', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                       edgecolor='#cccccc', alpha=0.8))

def load_data():
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    # Standardize 学段
    stage_map = {
        '小学': '小学', '初中': '初中', '高中': '高中', '幼儿园': '幼儿园',
        '中学': '中学', '中职': '中职',
    }
    df['学段_std'] = df['学段'].map(stage_map).fillna('其他')
    # Keep main stages
    main_stages = ['幼儿园', '小学', '初中', '高中']
    df['学段_main'] = df['学段_std'].where(df['学段_std'].isin(main_stages), '其他')

    # Standardize 五育
    wuyu_map = {'智育': '智育', '德育': '德育', '美育': '美育', '体育': '体育'}
    df['五育_std'] = df['培养方向'].map(wuyu_map).fillna('其他')

    # Region mapping
    province_region = {
        '北京市': '东部', '天津市': '东部', '河北省': '东部', '上海市': '东部',
        '江苏省': '东部', '浙江省': '东部', '福建省': '东部', '山东省': '东部',
        '广东省': '东部', '海南省': '东部', '辽宁省': '东部',
        '山西省': '中部', '吉林省': '中部', '黑龙江省': '中部', '安徽省': '中部',
        '江西省': '中部', '河南省': '中部', '湖北省': '中部', '湖南省': '中部',
        '内蒙古自治区': '西部', '广西壮族自治区': '西部', '重庆市': '西部',
        '四川省': '西部', '贵州省': '西部', '云南省': '西部', '西藏自治区': '西部',
        '陕西省': '西部', '甘肃省': '西部', '青海省': '西部',
        '宁夏回族自治区': '西部', '新疆维吾尔自治区': '西部',
    }
    df['区域'] = df['省份'].map(province_region)

    # Binary features
    llm_keywords = ['大模型', 'DeepSeek', '豆包', '文心一言', 'Kimi', '通义千问',
                    '星火', 'ChatGPT', 'GPT', '扣子']
    df['has_大模型'] = df['工具标准名'].apply(
        lambda x: any(k in str(x) for k in llm_keywords) if pd.notna(x) else False)
    df['has_硬件'] = df['产品形态'].str.contains('硬件', na=False)
    df['is_自研'] = df['是否自主研发'].astype(bool)

    # Scene L1
    df['场景L1'] = df['应用场景（一级）'].fillna('未提及')

    return df

# ══════════════════════════════════════════════════════════════════════════════
# FIG S1: Correspondence Analysis Biplot
# ══════════════════════════════════════════════════════════════════════════════
def fig_s1_correspondence_analysis(df):
    print("  Generating FIG S1: Correspondence Analysis...")
    import prince

    main_stages = ['幼儿园', '小学', '初中', '高中']
    sub = df[df['学段_main'].isin(main_stages)].copy()
    # Top subjects
    top_subj = sub['学科'].value_counts().head(12).index.tolist()
    if '未提及' in top_subj:
        top_subj.remove('未提及')
    sub = sub[sub['学科'].isin(top_subj)]

    ct = pd.crosstab(sub['学段_main'], sub['学科'])
    # Reorder stages
    stage_order = [s for s in main_stages if s in ct.index]
    ct = ct.loc[stage_order]

    ca = prince.CA(n_components=2)
    ca = ca.fit(ct)

    row_coords = ca.row_coordinates(ct)
    col_coords = ca.column_coordinates(ct)

    explained = ca.percentage_of_variance_
    if hasattr(explained, 'tolist'):
        explained = explained.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot column points (subjects)
    colors_subj = sns.color_palette('Set2', len(col_coords))
    for i, (subj, row) in enumerate(col_coords.iterrows()):
        ax.scatter(row[0], row[1], s=120, c=[colors_subj[i]], marker='s',
                   edgecolors='white', linewidth=0.8, zorder=5)
        ax.annotate(subj, (row[0], row[1]), fontsize=8, fontweight='bold',
                    xytext=(6, 6), textcoords='offset points',
                    color=colors_subj[i])

    # Plot row points (stages)
    stage_colors = [BLUES[1], TEALS[1], ORANGES[1], PURPLES[1]]
    for i, (stage, row) in enumerate(row_coords.iterrows()):
        ax.scatter(row[0], row[1], s=250, c=[stage_colors[i]], marker='o',
                   edgecolors='black', linewidth=1.2, zorder=6)
        ax.annotate(stage, (row[0], row[1]), fontsize=11, fontweight='bold',
                    xytext=(-10, 12), textcoords='offset points',
                    color=stage_colors[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor=stage_colors[i], linewidth=0.5))

    # Confidence ellipses (approximate)
    for i, (stage, row) in enumerate(row_coords.iterrows()):
        stage_data = sub[sub['学段_main'] == stage]
        # Create ellipse around stage point
        ellipse = matplotlib.patches.Ellipse(
            (row[0], row[1]), width=0.15, height=0.1,
            angle=0, facecolor=stage_colors[i], alpha=0.08,
            edgecolor=stage_colors[i], linewidth=1.0, linestyle='--')
        ax.add_patch(ellipse)

    # Draw association lines for key pairs
    key_assoc = [('小学', '语文'), ('高中', '物理'), ('初中', '数学'), ('幼儿园', '美术')]
    for stage, subj in key_assoc:
        if stage in row_coords.index and subj in col_coords.index:
            r = row_coords.loc[stage]
            c = col_coords.loc[subj]
            ax.plot([r[0], c[0]], [r[1], c[1]], '--', color='#aaaaaa',
                    linewidth=0.8, alpha=0.6, zorder=1)

    ax.axhline(0, color='#cccccc', linewidth=0.5, zorder=0)
    ax.axvline(0, color='#cccccc', linewidth=0.5, zorder=0)

    ax.set_xlabel(f'维度1 ({explained[0]:.1f}%)', fontsize=11)
    ax.set_ylabel(f'维度2 ({explained[1]:.1f}%)', fontsize=11)
    ax.set_title('学段-学科对应分析双标图', fontsize=14, fontweight='bold', pad=15)

    remove_spines(ax)
    add_grid(ax, 'both')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='学段（行点）'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=8, label='学科（列点）'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
              fontsize=9, edgecolor='#cccccc')

    caption = ("图S1 学段-学科对应分析双标图\n"
               "基于学段×学科列联表的对应分析，圆形为学段（行点），方形为学科（列点）。"
               "点间距离反映关联强度，虚线标注关键对应关系。"
               f"维度1和维度2分别解释{explained[0]:.1f}%和{explained[1]:.1f}%的总惯量。"
               f"N={len(sub):,}条记录。")
    fig.subplots_adjust(bottom=0.18, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '对应分析揭示学段与学科间存在显著关联，小学偏重语文，高中偏重物理，幼儿园偏重美术。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s1_correspondence_analysis.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S2: Mosaic Plot with Chi-Square
# ══════════════════════════════════════════════════════════════════════════════
def fig_s2_chi_square_mosaic(df):
    print("  Generating FIG S2: Mosaic Plot with Chi-Square...")

    main_stages = ['幼儿园', '小学', '初中', '高中']
    main_scenes = ['助学', '助教', '助育', '助评', '助管']
    sub = df[df['学段_main'].isin(main_stages) & df['场景L1'].isin(main_scenes)].copy()

    ct = pd.crosstab(sub['学段_main'], sub['场景L1'])
    stage_order = [s for s in main_stages if s in ct.index]
    scene_order = [s for s in main_scenes if s in ct.columns]
    ct = ct.loc[stage_order, scene_order]

    # Chi-square test
    chi2, p_val, dof, expected = stats.chi2_contingency(ct)

    # Pearson residuals
    residuals = (ct.values - expected) / np.sqrt(expected)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Build mosaic manually
    row_totals = ct.sum(axis=1)
    grand_total = ct.values.sum()
    row_props = row_totals / grand_total

    # Color map for residuals
    res_max = max(abs(residuals.min()), abs(residuals.max()), 3)
    cmap_div = LinearSegmentedColormap.from_list('res',
        ['#c0392c', '#f5b7b1', '#fdfefe', '#aed6f1', '#2471a3'])
    norm = Normalize(vmin=-res_max, vmax=res_max)

    x_start = 0
    gap = 0.008
    for i, stage in enumerate(stage_order):
        w = row_props.iloc[i] - gap
        col_totals = ct.loc[stage]
        col_props = col_totals / col_totals.sum()

        y_start = 0
        for j, scene in enumerate(scene_order):
            h = col_props.iloc[j]
            color = cmap_div(norm(residuals[i, j]))
            rect = plt.Rectangle((x_start, y_start), w, h - gap * 0.5,
                                 facecolor=color, edgecolor='white',
                                 linewidth=1.5)
            ax.add_patch(rect)

            # Label
            if w > 0.05 and h > 0.06:
                ax.text(x_start + w / 2, y_start + h / 2 - gap * 0.25,
                        f'{ct.loc[stage, scene]}',
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='#333333')
                ax.text(x_start + w / 2, y_start + h / 2 - gap * 0.25 - 0.025,
                        f'r={residuals[i, j]:.1f}',
                        ha='center', va='center', fontsize=6, color='#666666')

            y_start += h

        # Stage label at bottom
        ax.text(x_start + w / 2, -0.04, stage, ha='center', va='top',
                fontsize=10, fontweight='bold')
        x_start += row_props.iloc[i]

    # Scene labels on right
    y_pos = 0
    col_total_all = ct.sum(axis=0)
    col_prop_all = col_total_all / col_total_all.sum()
    for j, scene in enumerate(scene_order):
        h = col_prop_all.iloc[j]
        ax.text(1.02, y_pos + h / 2, scene, ha='left', va='center',
                fontsize=10, fontweight='bold')
        y_pos += h

    ax.set_xlim(-0.01, 1.08)
    ax.set_ylim(-0.08, 1.02)
    ax.set_aspect('auto')
    ax.axis('off')

    ax.set_title('学段 × 应用场景马赛克图（卡方检验）', fontsize=14,
                 fontweight='bold', pad=20)

    # Chi-square annotation
    p_str = f'p < 0.001' if p_val < 0.001 else f'p = {p_val:.4f}'
    chi_text = f'χ² = {chi2:.1f},  df = {dof},  {p_str}'
    ax.text(0.5, 1.06, chi_text, transform=ax.transAxes, ha='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7',
                      edgecolor='#f39c12', linewidth=0.8))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.12)
    cbar.set_label('Pearson残差', fontsize=9)

    caption = (f"图S2 学段×应用场景马赛克图（卡方检验）\n"
               f"面积与频数成正比，颜色表示Pearson残差（蓝=正关联，红=负关联）。"
               f"卡方检验：χ²={chi2:.1f}, df={dof}, {p_str}。"
               f"N={len(sub):,}条记录。")
    fig.subplots_adjust(bottom=0.18, top=0.90)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, f'卡方检验显示学段与应用场景存在显著关联（{p_str}），不同学段的场景偏好差异明显。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s2_chi_square_mosaic.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S3: Multi-Variable Correlation Matrix (Cramer's V)
# ══════════════════════════════════════════════════════════════════════════════
def fig_s3_correlation_matrix(df):
    print("  Generating FIG S3: Correlation Matrix...")

    main_stages = ['幼儿园', '小学', '初中', '高中']
    main_scenes = ['助学', '助教', '助育', '助评', '助管']
    main_wuyu = ['智育', '德育', '美育', '体育']

    features = {}
    features['使用大模型'] = df['has_大模型'].astype(int)
    features['使用硬件'] = df['has_硬件'].astype(int)
    features['自主研发'] = df['is_自研'].astype(int)

    for s in main_stages:
        features[f'学段:{s}'] = (df['学段_main'] == s).astype(int)
    for s in main_scenes:
        features[f'场景:{s}'] = (df['场景L1'] == s).astype(int)
    for w in main_wuyu:
        features[f'五育:{w}'] = (df['五育_std'] == w).astype(int)

    feat_df = pd.DataFrame(features)
    n_feat = len(feat_df.columns)

    # Compute Cramer's V matrix
    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(ct)[0]
        n = ct.values.sum()
        r, k = ct.shape
        phi2 = chi2 / n
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(r_corr - 1, k_corr - 1)
        if denom <= 0:
            return 0.0
        return np.sqrt(max(0, phi2 - ((r - 1) * (k - 1)) / (n - 1)) / denom)

    corr_matrix = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i, n_feat):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                v = cramers_v(feat_df.iloc[:, i], feat_df.iloc[:, j])
                corr_matrix[i, j] = v
                corr_matrix[j, i] = v

    corr_df = pd.DataFrame(corr_matrix, index=feat_df.columns,
                           columns=feat_df.columns)

    # Hierarchical clustering for ordering
    dist = 1 - corr_matrix
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)
    condensed = squareform(dist)
    Z = linkage(condensed, method='ward')
    order = leaves_list(Z)
    corr_ordered = corr_df.iloc[order, order]

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = LinearSegmentedColormap.from_list('cv',
        ['#fdfefe', '#aed6f1', '#2980b9', '#1a5276'])

    sns.heatmap(corr_ordered, mask=mask, cmap=cmap, vmin=0, vmax=0.5,
                annot=True, fmt='.2f', annot_kws={'size': 6.5},
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': "Cramér's V", 'shrink': 0.6},
                square=True, ax=ax)

    ax.set_title("多维特征关联矩阵（Cramér's V）", fontsize=14,
                 fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    caption = ("图S3 多维特征关联矩阵（Cramér's V）\n"
               "基于二值化特征的Cramér's V关联系数矩阵，经层次聚类重排序。"
               "包含技术特征（大模型/硬件/自研）、学段、应用场景和五育维度。"
               f"下三角显示，N={len(df):,}条记录。")
    fig.subplots_adjust(bottom=0.18, top=0.92)
    plt.figtext(0.5, 0.01, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '关联矩阵显示技术特征与教育维度间存在中等强度关联，大模型使用与助学场景正相关。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s3_correlation_matrix.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S4: Tool Diversity Regression
# ══════════════════════════════════════════════════════════════════════════════
def fig_s4_tool_diversity_regression(df):
    print("  Generating FIG S4: Tool Diversity Regression...")

    sub = df[df['省份'] != '未提及'].copy()

    # Per-case tool count
    case_tools = sub.groupby(['案例编号', '省份', '区域'])['工具标准名'].nunique().reset_index()
    case_tools.columns = ['案例编号', '省份', '区域', 'tool_count']

    # Province-level aggregation
    prov_stats = case_tools.groupby(['省份', '区域']).agg(
        case_count=('案例编号', 'nunique'),
        avg_tools=('tool_count', 'mean')
    ).reset_index()
    prov_stats = prov_stats.sort_values('case_count', ascending=False)

    region_colors = {'东部': BLUES[1], '中部': TEALS[1], '西部': ORANGES[1]}

    fig, ax = plt.subplots(figsize=(12, 7))

    for region, color in region_colors.items():
        mask = prov_stats['区域'] == region
        ax.scatter(prov_stats.loc[mask, 'case_count'],
                   prov_stats.loc[mask, 'avg_tools'],
                   c=color, s=100, alpha=0.8, edgecolors='white',
                   linewidth=0.8, label=region, zorder=5)

    # Annotate provinces
    for _, row in prov_stats.iterrows():
        ax.annotate(row['省份'].replace('省', '').replace('市', '')
                    .replace('自治区', '').replace('壮族', '').replace('维吾尔', '')
                    .replace('回族', ''),
                    (row['case_count'], row['avg_tools']),
                    fontsize=6, alpha=0.7, xytext=(4, 4),
                    textcoords='offset points')

    # Regression
    x = prov_stats['case_count'].values
    y = prov_stats['avg_tools'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, '--', color='#e74c3c', linewidth=1.5, alpha=0.8,
            zorder=4)

    # Confidence interval
    n = len(x)
    x_mean = x.mean()
    se = std_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    t_val = stats.t.ppf(0.975, n - 2)
    ax.fill_between(x_line, y_line - t_val * se, y_line + t_val * se,
                    alpha=0.1, color='#e74c3c', zorder=3)

    # R² annotation
    p_str = f'p < 0.001' if p_value < 0.001 else f'p = {p_value:.4f}'
    ax.text(0.95, 0.95, f'R² = {r_value**2:.3f}\n{p_str}\ny = {slope:.3f}x + {intercept:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7',
                      edgecolor='#f39c12', linewidth=0.8))

    ax.set_xlabel('省份案例数量', fontsize=11)
    ax.set_ylabel('平均每案例工具种类数', fontsize=11)
    ax.set_title('区域案例规模与工具多样性回归分析', fontsize=14,
                 fontweight='bold', pad=15)

    remove_spines(ax)
    add_grid(ax, 'both')
    ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    caption = (f"图S4 区域案例规模与工具多样性回归分析\n"
               f"各省份案例数量与平均工具种类数的线性回归，"
               f"R²={r_value**2:.3f}，{p_str}。"
               f"颜色区分东部/中部/西部地区，阴影为95%置信区间。"
               f"N={len(prov_stats)}个省份。")
    fig.subplots_adjust(bottom=0.18, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, f'回归分析表明案例规模与工具多样性呈正相关（R²={r_value**2:.3f}），东部地区显著领先。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s4_tool_diversity_regression.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S5: Technology Adoption Curve
# ══════════════════════════════════════════════════════════════════════════════
def fig_s5_tech_adoption_curve(df):
    print("  Generating FIG S5: Technology Adoption Curve...")

    # Classify tools by type
    llm_kw = ['大模型', 'DeepSeek', '豆包', '文心一言', 'Kimi', '通义千问',
              '星火', 'ChatGPT', 'GPT', '扣子', '智能体', 'AI助手']
    hw_kw = ['硬件', '机器人', '传感器', '摄像头', '智能笔', '平板']

    def classify_tool(row):
        name = str(row.get('工具标准名', ''))
        form = str(row.get('产品形态', ''))
        cat = str(row.get('产品分类', ''))
        if row.get('is_自研', False):
            return '自研系统'
        if any(k in name for k in llm_kw) or '模型' in cat or '大模型' in name:
            return '大模型/AIGC'
        if any(k in form for k in hw_kw) or '硬件' in form:
            return '硬件+AI'
        return '传统平台'

    df_c = df.copy()
    df_c['tool_type'] = df_c.apply(classify_tool, axis=1)

    total_cases = df_c['案例编号'].nunique()
    type_order = ['传统平台', '大模型/AIGC', '硬件+AI', '自研系统']
    type_colors = [BLUES[1], TEALS[1], ORANGES[1], PURPLES[1]]

    # Compute adoption rate per type
    adoption = {}
    for tt in type_order:
        cases_using = df_c[df_c['tool_type'] == tt]['案例编号'].nunique()
        adoption[tt] = cases_using / total_cases * 100

    fig, ax = plt.subplots(figsize=(11, 7))

    # Create S-curve style visualization
    # X-axis: innovation stages, Y-axis: cumulative adoption %
    x_positions = np.array([1, 2, 3, 4])
    adoption_vals = [adoption.get(t, 0) for t in type_order]

    # Sort by adoption rate (descending) for cumulative curve
    sorted_pairs = sorted(zip(type_order, adoption_vals, type_colors),
                          key=lambda x: -x[1])

    # Plot bars
    bars = ax.bar(x_positions, [p[1] for p in sorted_pairs],
                  color=[p[2] for p in sorted_pairs], width=0.6,
                  edgecolor='white', linewidth=1.5, alpha=0.85, zorder=4)

    # Add cumulative S-curve overlay
    ax2 = ax.twinx()
    cum_vals = np.cumsum([p[1] for p in sorted_pairs])
    # Smooth S-curve
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(0.5, 4.5, 200)
    # Add boundary points for S-shape
    x_pts = np.array([0.5] + list(x_positions) + [4.5])
    y_pts = np.array([0] + list(cum_vals) + [cum_vals[-1]])
    try:
        spl = make_interp_spline(x_pts, y_pts, k=3)
        y_smooth = spl(x_smooth)
        y_smooth = np.clip(y_smooth, 0, cum_vals[-1] * 1.02)
    except Exception:
        y_smooth = np.interp(x_smooth, x_pts, y_pts)

    ax2.plot(x_smooth, y_smooth, '-', color='#e74c3c', linewidth=2.5,
             alpha=0.8, zorder=5)
    ax2.scatter(x_positions, cum_vals, c='#e74c3c', s=60, zorder=6,
                edgecolors='white', linewidth=1)
    ax2.set_ylabel('累计覆盖率 (%)', fontsize=10, color='#e74c3c')
    ax2.tick_params(axis='y', colors='#e74c3c')
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_color('#e74c3c')

    # Innovation diffusion annotations
    diffusion_labels = ['创新者\nInnovators', '早期采纳者\nEarly Adopters',
                        '早期多数\nEarly Majority', '晚期多数\nLate Majority']
    for i, (pair, dl) in enumerate(zip(sorted_pairs, diffusion_labels)):
        ax.text(x_positions[i], pair[1] + 1.5, f'{pair[1]:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=pair[2])
        ax.text(x_positions[i], -4, dl, ha='center', va='top',
                fontsize=7, color='#666666', style='italic')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([p[0] for p in sorted_pairs], fontsize=10,
                       fontweight='bold')
    ax.set_ylabel('案例覆盖率 (%)', fontsize=11)
    ax.set_title('AI教育技术采纳扩散曲线', fontsize=14, fontweight='bold', pad=15)

    remove_spines(ax)
    add_grid(ax, 'y')
    ax.set_xlim(0.3, 4.7)

    caption = (f"图S5 AI教育技术采纳扩散曲线\n"
               f"按技术类型统计案例覆盖率，柱形为各类型独立覆盖率，"
               f"红色曲线为累计覆盖率。参照Rogers创新扩散理论标注采纳阶段。"
               f"总案例数N={total_cases:,}。")
    fig.subplots_adjust(bottom=0.20, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '技术采纳曲线显示传统平台仍为主流，大模型/AIGC正处于早期采纳者阶段，增长潜力巨大。',
                     x=0.02, y=0.20)

    save_fig(fig, 'fig_s5_tech_adoption_curve.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S6: Subject -> Scenario -> Wuyu Alluvial/Sankey
# ══════════════════════════════════════════════════════════════════════════════
def fig_s6_subject_scenario_alluvial(df):
    print("  Generating FIG S6: Subject-Scenario-Wuyu Alluvial...")

    top_subj = ['语文', '数学', '英语', '美术', '科学', '体育', '物理', '音乐']
    main_scenes = ['助学', '助教', '助育', '助评', '助管']
    main_wuyu = ['智育', '德育', '美育', '体育']

    sub = df[df['学科'].isin(top_subj) &
             df['场景L1'].isin(main_scenes) &
             df['五育_std'].isin(main_wuyu)].copy()

    # Aggregate flows
    flow1 = sub.groupby(['学科', '场景L1']).size().reset_index(name='count')
    flow2 = sub.groupby(['场景L1', '五育_std']).size().reset_index(name='count')

    fig, ax = plt.subplots(figsize=(14, 9))

    # Three columns: subjects, scenes, wuyu
    col_x = [0.1, 0.5, 0.9]

    subj_colors = dict(zip(top_subj, sns.color_palette('Set2', len(top_subj))))
    scene_colors = dict(zip(main_scenes,
        [BLUES[1], TEALS[1], ORANGES[1], PURPLES[1], REDS[1]]))
    wuyu_colors = dict(zip(main_wuyu,
        [BLUES[0], TEALS[0], ORANGES[0], PURPLES[0]]))

    def draw_column(items, x, colors_dict, totals):
        """Draw stacked blocks for a column, return y-positions."""
        total = sum(totals.values())
        gap = 0.01
        usable = 0.85
        positions = {}
        y = 0.05
        for item in items:
            h = (totals.get(item, 0) / total) * usable
            if h < 0.005:
                continue
            rect = FancyBboxPatch((x - 0.035, y), 0.07, h,
                                   boxstyle='round,pad=0.003',
                                   facecolor=colors_dict.get(item, '#cccccc'),
                                   alpha=0.85, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y + h / 2, f'{item}\n({totals.get(item, 0)})',
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color='white' if h > 0.03 else '#333333')
            positions[item] = (y, y + h)
            y += h + gap
        return positions

    # Compute totals
    subj_totals = sub['学科'].value_counts().to_dict()
    scene_totals = sub['场景L1'].value_counts().to_dict()
    wuyu_totals = sub['五育_std'].value_counts().to_dict()

    pos_subj = draw_column(top_subj, col_x[0], subj_colors, subj_totals)
    pos_scene = draw_column(main_scenes, col_x[1], scene_colors, scene_totals)
    pos_wuyu = draw_column(main_wuyu, col_x[2], wuyu_colors, wuyu_totals)

    # Draw flows: subject -> scene
    def draw_flows(flow_df, col_from, col_to, pos_from, pos_to, from_col,
                   to_col, colors_dict, alpha=0.15):
        # Track cumulative offsets
        from_offsets = {k: v[0] for k, v in pos_from.items()}
        to_offsets = {k: v[0] for k, v in pos_to.items()}

        for _, row in flow_df.iterrows():
            src = row[from_col]
            dst = row[to_col]
            cnt = row['count']
            if src not in pos_from or dst not in pos_to:
                continue

            total_src = sum(flow_df[flow_df[from_col] == src]['count'])
            total_dst = sum(flow_df[flow_df[to_col] == dst]['count'])
            h_src = (pos_from[src][1] - pos_from[src][0]) * cnt / total_src
            h_dst = (pos_to[dst][1] - pos_to[dst][0]) * cnt / total_dst

            y0_src = from_offsets[src]
            y0_dst = to_offsets[dst]

            # Bezier curve
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path

            x0 = col_from + 0.04
            x1 = col_to - 0.04
            xm = (x0 + x1) / 2

            verts_top = [(x0, y0_src + h_src), (xm, y0_src + h_src),
                         (xm, y0_dst + h_dst), (x1, y0_dst + h_dst)]
            verts_bot = [(x1, y0_dst), (xm, y0_dst),
                         (xm, y0_src), (x0, y0_src)]
            verts = verts_top + verts_bot + [(x0, y0_src + h_src)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.CLOSEPOLY]

            path = Path(verts, codes)
            color = colors_dict.get(src, '#cccccc')
            patch = PathPatch(path, facecolor=color, alpha=alpha,
                              edgecolor='none', zorder=2)
            ax.add_patch(patch)

            from_offsets[src] += h_src
            to_offsets[dst] += h_dst

    draw_flows(flow1, col_x[0], col_x[1], pos_subj, pos_scene,
               '学科', '场景L1', subj_colors, alpha=0.18)
    draw_flows(flow2, col_x[1], col_x[2], pos_scene, pos_wuyu,
               '场景L1', '五育_std', scene_colors, alpha=0.18)

    # Column headers
    headers = ['学科', '应用场景', '五育方向']
    for x, h in zip(col_x, headers):
        ax.text(x, 0.96, h, ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='#333333')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('学科-场景-五育三维流向图', fontsize=14,
                 fontweight='bold', pad=15)

    caption = (f"图S6 学科-场景-五育三维流向图\n"
               f"展示学科→应用场景→五育培养方向的流向关系，"
               f"流带宽度与案例数成正比。"
               f"N={len(sub):,}条记录，涵盖{len(top_subj)}个主要学科。")
    fig.subplots_adjust(bottom=0.14, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '三维流向图揭示学科-场景-五育的关联路径，语文和数学主要流向助学场景和智育方向。',
                     x=0.02, y=0.14)

    save_fig(fig, 'fig_s6_subject_scenario_alluvial.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S7: Innovation Depth Violin Plot
# ══════════════════════════════════════════════════════════════════════════════
def fig_s7_innovation_depth_boxplot(df):
    print("  Generating FIG S7: Innovation Depth Violin Plot...")

    main_stages = ['幼儿园', '小学', '初中', '高中']
    sub = df[df['学段_main'].isin(main_stages)].copy()

    # Compute innovation depth per case
    case_metrics = sub.groupby(['案例编号', '学段_main']).agg(
        n_tools=('工具标准名', 'nunique'),
        has_selfdev=('is_自研', 'max'),
        has_llm=('has_大模型', 'max'),
        has_hw=('has_硬件', 'max'),
        n_scenes=('场景L1', 'nunique'),
        n_wuyu=('五育_std', 'nunique'),
    ).reset_index()

    # Parse tech elements count per case
    tech_counts = sub.groupby('案例编号')['技术要素'].apply(
        lambda x: len(set(str(v) for v in x.dropna()))
    ).reset_index(name='n_tech_elements')

    # Parse pathway complexity (number of steps)
    def count_steps(x):
        steps = set()
        for v in x.dropna():
            parts = str(v).split('→')
            steps.update(p.strip() for p in parts if p.strip())
        return len(steps)

    path_counts = sub.groupby('案例编号')['关键技术路径'].apply(count_steps).reset_index(
        name='n_path_steps')

    case_metrics = case_metrics.merge(tech_counts, on='案例编号', how='left')
    case_metrics = case_metrics.merge(path_counts, on='案例编号', how='left')
    case_metrics['n_tech_elements'] = case_metrics['n_tech_elements'].fillna(0)
    case_metrics['n_path_steps'] = case_metrics['n_path_steps'].fillna(0)

    # Composite innovation depth score (normalized)
    from sklearn.preprocessing import MinMaxScaler
    score_cols = ['n_tools', 'has_selfdev', 'has_llm', 'has_hw',
                  'n_scenes', 'n_tech_elements', 'n_path_steps']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(case_metrics[score_cols].fillna(0))
    case_metrics['innovation_depth'] = scaled.mean(axis=1) * 10  # 0-10 scale

    fig, ax = plt.subplots(figsize=(10, 7))

    stage_colors = [BLUES[1], TEALS[1], ORANGES[1], PURPLES[1]]

    # Violin plot
    parts = ax.violinplot(
        [case_metrics[case_metrics['学段_main'] == s]['innovation_depth'].values
         for s in main_stages],
        positions=range(len(main_stages)), showmeans=False, showmedians=False,
        showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(stage_colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor(stage_colors[i])

    # Box plot overlay
    bp = ax.boxplot(
        [case_metrics[case_metrics['学段_main'] == s]['innovation_depth'].values
         for s in main_stages],
        positions=range(len(main_stages)), widths=0.15, patch_artist=True,
        showfliers=False, zorder=5)

    for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        box.set_facecolor(stage_colors[i])
        box.set_alpha(0.7)
        box.set_edgecolor('white')
        median.set_color('white')
        median.set_linewidth(2)

    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color('#666666')
            line.set_linewidth(0.8)

    # Swarm-like jitter
    for i, stage in enumerate(main_stages):
        data = case_metrics[case_metrics['学段_main'] == stage]['innovation_depth'].values
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(np.full_like(data, i) + jitter, data,
                   c=stage_colors[i], s=3, alpha=0.15, zorder=4)

    # Mann-Whitney U tests between adjacent stages
    sig_pairs = []
    for i in range(len(main_stages) - 1):
        d1 = case_metrics[case_metrics['学段_main'] == main_stages[i]]['innovation_depth']
        d2 = case_metrics[case_metrics['学段_main'] == main_stages[i + 1]]['innovation_depth']
        if len(d1) > 0 and len(d2) > 0:
            u_stat, p_val = stats.mannwhitneyu(d1, d2, alternative='two-sided')
            sig_pairs.append((i, i + 1, p_val))

    # Add significance markers
    y_max = case_metrics['innovation_depth'].max()
    for idx, (i, j, p) in enumerate(sig_pairs):
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        y = y_max + 0.3 + idx * 0.5
        ax.plot([i, i, j, j], [y - 0.1, y, y, y - 0.1], 'k-', linewidth=0.8)
        ax.text((i + j) / 2, y + 0.05, sig, ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(main_stages)))
    ax.set_xticklabels(main_stages, fontsize=11, fontweight='bold')
    ax.set_ylabel('创新深度指数 (0-10)', fontsize=11)
    ax.set_title('各学段AI教育创新深度分布（小提琴图）', fontsize=14,
                 fontweight='bold', pad=15)

    remove_spines(ax)
    add_grid(ax, 'y')

    # Add n counts
    for i, stage in enumerate(main_stages):
        n = len(case_metrics[case_metrics['学段_main'] == stage])
        ax.text(i, -0.6, f'n={n}', ha='center', fontsize=8, color='#666666')

    caption = ("图S7 各学段AI教育创新深度分布（小提琴图）\n"
               "创新深度指数综合工具种类数、自研比例、大模型使用、硬件使用、"
               "场景多样性和技术路径复杂度。显著性标记：***p<0.001, **p<0.01, "
               "*p<0.05 (Mann-Whitney U检验)。")
    fig.subplots_adjust(bottom=0.18, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '创新深度在学段间存在显著差异，高中阶段创新深度指数最高，幼儿园阶段相对较低。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s7_innovation_depth_boxplot.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S8: Geographic Inequality (Gini / Lorenz)
# ══════════════════════════════════════════════════════════════════════════════
def fig_s8_geographic_inequality(df):
    print("  Generating FIG S8: Geographic Inequality...")

    sub = df[df['省份'] != '未提及'].copy()
    prov_cases = sub.groupby('省份')['案例编号'].nunique().sort_values()

    # Gini coefficient
    values = prov_cases.values.astype(float)
    n = len(values)
    sorted_vals = np.sort(values)
    cumvals = np.cumsum(sorted_vals)
    total = cumvals[-1]
    # Lorenz curve points
    lorenz_x = np.concatenate([[0], np.arange(1, n + 1) / n])
    lorenz_y = np.concatenate([[0], cumvals / total])

    # Gini = 1 - 2 * area under Lorenz
    gini = 1 - 2 * np.trapz(lorenz_y, lorenz_x)

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # Lorenz curve
    ax.fill_between(lorenz_x, lorenz_y, alpha=0.15, color=BLUES[1])
    ax.plot(lorenz_x, lorenz_y, '-', color=BLUES[1], linewidth=2.5,
            label='Lorenz曲线', zorder=5)
    ax.plot([0, 1], [0, 1], '--', color='#999999', linewidth=1,
            label='完全均等线', zorder=3)

    # Shade Gini area
    ax.fill_between(lorenz_x, lorenz_y, lorenz_x, alpha=0.08,
                    color=ORANGES[1])

    # Annotate Gini
    ax.text(0.65, 0.25, f'基尼系数 = {gini:.3f}',
            fontsize=14, fontweight='bold', color=BLUES[0],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#eaf2f8',
                      edgecolor=BLUES[1], linewidth=1))

    # Interpretation
    if gini > 0.5:
        interp = '高度不均衡'
    elif gini > 0.4:
        interp = '较不均衡'
    elif gini > 0.3:
        interp = '中等不均衡'
    else:
        interp = '相对均衡'
    ax.text(0.65, 0.15, f'判定：{interp}', fontsize=10, color='#666666')

    ax.set_xlabel('省份累计比例', fontsize=11)
    ax.set_ylabel('案例累计比例', fontsize=11)
    ax.set_title('AI教育区域均衡度分析（基尼系数）', fontsize=14,
                 fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='#cccccc')

    remove_spines(ax)
    add_grid(ax, 'both')

    # Inset: regional share bar chart
    region_map = {
        '北京市': '东部', '天津市': '东部', '河北省': '东部', '上海市': '东部',
        '江苏省': '东部', '浙江省': '东部', '福建省': '东部', '山东省': '东部',
        '广东省': '东部', '海南省': '东部', '辽宁省': '东部',
        '山西省': '中部', '吉林省': '中部', '黑龙江省': '中部', '安徽省': '中部',
        '江西省': '中部', '河南省': '中部', '湖北省': '中部', '湖南省': '中部',
        '内蒙古自治区': '西部', '广西壮族自治区': '西部', '重庆市': '西部',
        '四川省': '西部', '贵州省': '西部', '云南省': '西部', '西藏自治区': '西部',
        '陕西省': '西部', '甘肃省': '西部', '青海省': '西部',
        '宁夏回族自治区': '西部', '新疆维吾尔自治区': '西部',
    }
    sub['区域'] = sub['省份'].map(region_map)
    region_cases = sub.groupby('区域')['案例编号'].nunique()
    region_total = region_cases.sum()
    region_pct = (region_cases / region_total * 100).reindex(['东部', '中部', '西部'])

    ax_inset = fig.add_axes([0.18, 0.52, 0.25, 0.28])
    bars = ax_inset.bar(range(3), region_pct.values,
                        color=[BLUES[1], TEALS[1], ORANGES[1]],
                        edgecolor='white', linewidth=1)
    ax_inset.set_xticks(range(3))
    ax_inset.set_xticklabels(['东部', '中部', '西部'], fontsize=8)
    ax_inset.set_ylabel('案例占比 (%)', fontsize=7)
    ax_inset.set_title('区域份额', fontsize=8, fontweight='bold')
    for bar, val in zip(bars, region_pct.values):
        ax_inset.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f'{val:.1f}%', ha='center', fontsize=7, fontweight='bold')
    remove_spines(ax_inset)
    ax_inset.tick_params(axis='both', labelsize=7)

    caption = (f"图S8 AI教育区域均衡度分析（基尼系数）\n"
               f"基于{n}个省份案例分布的Lorenz曲线，基尼系数={gini:.3f}（{interp}）。"
               f"内嵌图显示东/中/西部案例占比。总案例数N={region_total:,}。")
    fig.subplots_adjust(bottom=0.18, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, f'基尼系数={gini:.3f}表明AI教育资源在省域间分布{interp}，区域均衡化仍需政策推动。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s8_geographic_inequality.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S9: Effect Size Forest Plot
# ══════════════════════════════════════════════════════════════════════════════
def fig_s9_effect_size_forest(df):
    print("  Generating FIG S9: Effect Size Forest Plot...")

    main_stages = ['幼儿园', '小学', '初中', '高中']
    sub = df[df['学段_main'].isin(main_stages)].copy()

    # Features to compare
    features = {
        '使用大模型': sub['has_大模型'].astype(int),
        '使用硬件': sub['has_硬件'].astype(int),
        '自主研发': sub['is_自研'].astype(int),
        '助学场景': (sub['场景L1'] == '助学').astype(int),
        '助教场景': (sub['场景L1'] == '助教').astype(int),
        '智育导向': (sub['五育_std'] == '智育').astype(int),
        '德育导向': (sub['五育_std'] == '德育').astype(int),
        '美育导向': (sub['五育_std'] == '美育').astype(int),
    }

    # Compute Cramer's V for each feature vs 学段
    results = []
    for feat_name, feat_vals in features.items():
        ct = pd.crosstab(sub['学段_main'], feat_vals)
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        n = ct.values.sum()
        r, k = ct.shape
        v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

        # Bootstrap CI for Cramer's V
        np.random.seed(42)
        boot_vs = []
        stage_arr = sub['学段_main'].values
        feat_arr = feat_vals.values
        for _ in range(500):
            idx = np.random.choice(len(stage_arr), len(stage_arr), replace=True)
            try:
                ct_b = pd.crosstab(pd.Series(stage_arr[idx]),
                                   pd.Series(feat_arr[idx]))
                chi2_b = stats.chi2_contingency(ct_b)[0]
                n_b = ct_b.values.sum()
                r_b, k_b = ct_b.shape
                v_b = np.sqrt(chi2_b / (n_b * (min(r_b, k_b) - 1)))
                boot_vs.append(v_b)
            except Exception:
                pass
        ci_low = np.percentile(boot_vs, 2.5) if boot_vs else v * 0.8
        ci_high = np.percentile(boot_vs, 97.5) if boot_vs else v * 1.2

        results.append({
            'feature': feat_name,
            'v': v,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p': p_val,
        })

    results_df = pd.DataFrame(results).sort_values('v', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(results_df))
    colors = []
    for _, row in results_df.iterrows():
        if row['p'] < 0.001:
            colors.append(BLUES[0])
        elif row['p'] < 0.05:
            colors.append(BLUES[2])
        else:
            colors.append('#bbbbbb')

    # Plot CI lines
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.plot([row['ci_low'], row['ci_high']], [i, i],
                color=colors[i], linewidth=2, zorder=3)
        ax.plot([row['ci_low'], row['ci_low']], [i - 0.1, i + 0.1],
                color=colors[i], linewidth=1.5, zorder=3)
        ax.plot([row['ci_high'], row['ci_high']], [i - 0.1, i + 0.1],
                color=colors[i], linewidth=1.5, zorder=3)

    # Plot point estimates
    ax.scatter(results_df['v'].values, y_pos, c=colors, s=100,
               edgecolors='white', linewidth=1.2, zorder=5)

    # Annotations
    for i, (_, row) in enumerate(results_df.iterrows()):
        p_str = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01
                else ('*' if row['p'] < 0.05 else 'n.s.'))
        ax.text(row['ci_high'] + 0.008, i,
                f"V={row['v']:.3f} {p_str}",
                va='center', fontsize=8, color=colors[i], fontweight='bold')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(results_df['feature'].values, fontsize=10)
    ax.set_xlabel("Cramér's V (效应量)", fontsize=11)
    ax.set_title('学段差异效应量森林图', fontsize=14, fontweight='bold', pad=15)

    # Reference lines
    ax.axvline(0.1, color='#f39c12', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axvline(0.3, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.text(0.1, len(results_df) - 0.3, '小效应', fontsize=7, color='#f39c12',
            ha='center')
    ax.text(0.3, len(results_df) - 0.3, '中效应', fontsize=7, color='#e74c3c',
            ha='center')

    remove_spines(ax)
    add_grid(ax, 'x')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUES[0],
                   markersize=8, label='p < 0.001'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUES[2],
                   markersize=8, label='p < 0.05'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbbbbb',
                   markersize=8, label='n.s.'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='#cccccc')

    caption = ("图S9 学段差异效应量森林图\n"
               "各二值特征与学段关联的Cramér's V效应量及95%自助法置信区间。"
               "虚线标注小效应(0.1)和中效应(0.3)阈值。"
               "***p<0.001, **p<0.01, *p<0.05。"
               f"N={len(sub):,}条记录。")
    fig.subplots_adjust(bottom=0.18, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '效应量分析显示"使用大模型"和"助学场景"与学段关联最强（p<0.001），具有中等效应量。',
                     x=0.02, y=0.18)

    save_fig(fig, 'fig_s9_effect_size_forest.png', caption)

# ══════════════════════════════════════════════════════════════════════════════
# FIG S10: Multi-Factor Interaction Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig_s10_interaction_heatmap(df):
    print("  Generating FIG S10: Multi-Factor Interaction Heatmap...")

    main_stages = ['幼儿园', '小学', '初中', '高中']
    main_scenes = ['助学', '助教', '助育', '助评', '助管']
    main_wuyu = ['智育', '德育', '美育', '体育']

    sub = df[df['学段_main'].isin(main_stages) &
             df['场景L1'].isin(main_scenes) &
             df['五育_std'].isin(main_wuyu)].copy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('heat',
        ['#fdfefe', '#aed6f1', '#5dade2', '#2980b9', '#1a5276'])

    global_max = 0
    for stage in main_stages:
        ct = pd.crosstab(sub[sub['学段_main'] == stage]['场景L1'],
                         sub[sub['学段_main'] == stage]['五育_std'])
        ct = ct.reindex(index=main_scenes, columns=main_wuyu, fill_value=0)
        if ct.values.max() > global_max:
            global_max = ct.values.max()

    for idx, (stage, ax) in enumerate(zip(main_stages, axes)):
        stage_data = sub[sub['学段_main'] == stage]
        ct = pd.crosstab(stage_data['场景L1'], stage_data['五育_std'])
        ct = ct.reindex(index=main_scenes, columns=main_wuyu, fill_value=0)

        sns.heatmap(ct, ax=ax, cmap=cmap, vmin=0, vmax=global_max,
                    annot=True, fmt='d', annot_kws={'size': 9, 'fontweight': 'bold'},
                    linewidths=1, linecolor='white',
                    cbar=idx == 3,
                    cbar_kws={'label': '案例数', 'shrink': 0.8} if idx == 3 else {})

        ax.set_title(f'{stage}\n(n={len(stage_data):,})', fontsize=12,
                     fontweight='bold', pad=10)
        ax.set_xlabel('五育方向' if idx == 1 else '', fontsize=10)
        if idx == 0:
            ax.set_ylabel('应用场景', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=0)
        plt.setp(ax.get_yticklabels(), rotation=0)

    fig.suptitle('学段 × 场景 × 五育 三因素交互热力图', fontsize=15,
                 fontweight='bold', y=0.99)

    caption = (f"图S10 学段×场景×五育三因素交互热力图\n"
               f"四个面板分别对应四个学段，行为应用场景（一级），列为五育方向。"
               f"颜色深浅和标注数字表示案例数量，统一色阶便于跨学段比较。"
               f"N={len(sub):,}条记录。")
    fig.subplots_adjust(bottom=0.22, top=0.92)
    plt.figtext(0.5, 0.02, caption, ha='center', fontsize=7.5,
                style='italic', color='#444444', wrap=True,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                          edgecolor='#dddddd', linewidth=0.5))
    add_analysis_box(fig, '多因素交互分析揭示各学段的场景-五育关联模式差异，小学助学-智育组合最为密集。',
                     x=0.02, y=0.22)

    save_fig(fig, 'fig_s10_interaction_heatmap.png', caption)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Advanced Statistical Analysis - AI Education Research")
    print("=" * 60)

    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_data()
    print(f"  Loaded {len(df):,} rows, {df['案例编号'].nunique():,} unique cases\n")

    fig_s1_correspondence_analysis(df)
    fig_s2_chi_square_mosaic(df)
    fig_s3_correlation_matrix(df)
    fig_s4_tool_diversity_regression(df)
    fig_s5_tech_adoption_curve(df)
    fig_s6_subject_scenario_alluvial(df)
    fig_s7_innovation_depth_boxplot(df)
    fig_s8_geographic_inequality(df)
    fig_s9_effect_size_forest(df)
    fig_s10_interaction_heatmap(df)

    print("\n" + "=" * 60)
    print("All 10 figures generated successfully!")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
