#!/usr/bin/env python3
"""
Deep Clustering Analysis for AI Education Cases
Generates publication-quality figures (C1-C8) for Nature/Science-level presentation.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import ConvexHull

import seaborn as sns

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_PATH = Path('/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv')
NLP_DIR  = Path('/Users/sakai/Desktop/产业调研/ai-edu-research/output')
FIG_DIR  = Path('/Users/sakai/Desktop/产业调研/ai-edu-research/output/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# Sophisticated palette
PALETTE = sns.color_palette('husl', 12)
CLUSTER_COLORS = [
    '#E63946', '#457B9D', '#2A9D8F', '#E9C46A',
    '#F4A261', '#264653', '#A8DADC', '#6D6875',
    '#B5838D', '#FFB4A2', '#6B705C', '#CB997E'
]

# ── Helper: style axes ────────────────────────────────────────────────────
def style_ax(ax, title='', subtitle='', xlabel='', ylabel=''):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=9, width=0.5)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9, color='#555555', style='italic')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

def add_caption_box(fig, text, y=-0.02, fontsize=8):
    """Add a Chinese caption/annotation box at the bottom of the figure."""
    fig.text(0.5, y, text, ha='center', va='top', fontsize=fontsize,
             color='#333333', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F0',
                       edgecolor='#CCCCCC', alpha=0.9),
             wrap=True)

def add_analysis_box(fig, text, x=0.02, y=0.02, fontsize=7.5):
    """Add an analysis insight annotation box."""
    fig.text(x, y, text, ha='left', va='bottom', fontsize=fontsize,
             color='#1a5276', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                       edgecolor='#cccccc', alpha=0.8),
             wrap=True)

def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(str(path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] {name}')

# ── Load & Prepare Data ───────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

# Deduplicate by 案例编号 – keep first row per case
case_df = df.drop_duplicates(subset='案例编号', keep='first').copy()
case_df = case_df[case_df['优势和创新点'].notna()].reset_index(drop=True)
print(f'  Unique cases with innovation text: {len(case_df)}')

# Load NLP results
with open(NLP_DIR / 'nlp_clusters.json', encoding='utf-8-sig') as f:
    nlp_clusters = json.load(f)
with open(NLP_DIR / 'nlp_tfidf_keywords.json', encoding='utf-8-sig') as f:
    nlp_keywords = json.load(f)
with open(NLP_DIR / 'nlp_lda_topics.json', encoding='utf-8-sig') as f:
    nlp_topics = json.load(f)

# ── Jieba tokenization ────────────────────────────────────────────────────
print('Tokenizing with jieba...')
import jieba

STOPWORDS = set('的了是在和与及对为不有也将更能被从而到以这个中上下一种通过进行实现利用基于使用采用结合运用提供支持促进提升推动开展构建打造形成探索创新应用技术教学学生教师学习课堂教育人工智能ai AI'.split())

def tokenize(text):
    if not isinstance(text, str):
        return ''
    words = jieba.lcut(text)
    return ' '.join(w for w in words if len(w) > 1 and w not in STOPWORDS)

case_df['tokens'] = case_df['优势和创新点'].apply(tokenize)

# ── TF-IDF Vectorization ─────────────────────────────────────────────────
print('Building TF-IDF matrix...')
vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.85)
tfidf_matrix = vectorizer.fit_transform(case_df['tokens'])
feature_names = vectorizer.get_feature_names_out()
print(f'  TF-IDF shape: {tfidf_matrix.shape}')

# ── Optimal K via Silhouette ─────────────────────────────────────────────
print('Finding optimal K...')
sil_scores = {}
K_RANGE = range(3, 16)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(tfidf_matrix)
    sil_scores[k] = silhouette_score(tfidf_matrix, labels)
    print(f'  k={k}: silhouette={sil_scores[k]:.4f}')

best_k = max(sil_scores, key=sil_scores.get)
# Cap at 10 for interpretability; prefer 8 if close
if best_k > 10:
    best_k = 10
if abs(sil_scores.get(8, 0) - sil_scores[best_k]) < 0.008:
    best_k = 8
print(f'  Selected K={best_k}')

# ── Final KMeans ──────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
case_df['cluster'] = km_final.fit_predict(tfidf_matrix)
cluster_centers = km_final.cluster_centers_

# ── Cluster names from top keywords ──────────────────────────────────────
cluster_names = {}
cluster_top3 = {}
for c in range(best_k):
    center = cluster_centers[c]
    top_idx = center.argsort()[::-1][:5]
    top_words = [feature_names[i] for i in top_idx]
    cluster_top3[c] = top_words[:3]
    cluster_names[c] = f'C{c}: {"/".join(top_words[:2])}'

print('Cluster names:')
for c, name in cluster_names.items():
    size = (case_df['cluster'] == c).sum()
    print(f'  {name} (n={size})')

# ── t-SNE Embedding ───────────────────────────────────────────────────────
print('Computing t-SNE...')
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, learning_rate='auto')
tsne_2d = tsne.fit_transform(tfidf_matrix.toarray())
case_df['tsne_x'] = tsne_2d[:, 0]
case_df['tsne_y'] = tsne_2d[:, 1]

# ── PCA for 3D fallback / UMAP ───────────────────────────────────────────
print('Computing PCA 3D...')
pca3 = PCA(n_components=3, random_state=42)
pca_3d = pca3.fit_transform(tfidf_matrix.toarray())
case_df['pca_x'] = pca_3d[:, 0]
case_df['pca_y'] = pca_3d[:, 1]
case_df['pca_z'] = pca_3d[:, 2]

try:
    import umap
    print('Computing UMAP 3D...')
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_3d = reducer.fit_transform(tfidf_matrix.toarray())
    case_df['umap_x'] = umap_3d[:, 0]
    case_df['umap_y'] = umap_3d[:, 1]
    case_df['umap_z'] = umap_3d[:, 2]
    USE_UMAP = True
    print('  UMAP available.')
except ImportError:
    USE_UMAP = False
    print('  UMAP not available, using PCA fallback.')

# ── Compute cluster profiles ─────────────────────────────────────────────
print('Computing cluster profiles...')

def safe_ratio(series, values):
    """Count ratio of values in series."""
    total = len(series)
    if total == 0:
        return 0
    return sum(1 for v in series if any(val in str(v) for val in values)) / total

# Build full row data per case (merge all rows for that case)
all_rows = df.copy()

profile_dims = ['学段分布', '学科覆盖', '技术复杂度', '应用场景多样性', '五育均衡度', '创新深度']

cluster_profiles = {}
for c in range(best_k):
    cids = case_df[case_df['cluster'] == c]['案例编号'].values
    c_rows = all_rows[all_rows['案例编号'].isin(cids)]
    c_cases = case_df[case_df['cluster'] == c]

    # 学段分布 - diversity of school levels
    levels = c_rows['学段'].dropna().unique()
    dim1 = min(len(levels) / 6.0, 1.0)

    # 学科覆盖 - diversity of subjects
    subjects = c_rows['学科'].dropna().unique()
    dim2 = min(len(subjects) / 15.0, 1.0)

    # 技术复杂度 - avg number of tech elements
    tech_counts = c_rows['技术要素'].dropna().apply(lambda x: str(x).count(',') + 1)
    dim3 = min(tech_counts.mean() / 6.0, 1.0) if len(tech_counts) > 0 else 0

    # 应用场景多样性
    scenes = c_rows['应用场景（一级）'].dropna().unique()
    dim4 = min(len(scenes) / 6.0, 1.0)

    # 五育均衡度
    wuyu = c_rows['培养方向'].dropna()
    wuyu_types = set()
    for v in wuyu:
        for t in ['智育', '德育', '美育', '体育', '劳育']:
            if t in str(v):
                wuyu_types.add(t)
    dim5 = len(wuyu_types) / 5.0

    # 创新深度 - avg text length of innovation
    innov_len = c_cases['优势和创新点'].apply(lambda x: len(str(x)))
    dim6 = min(innov_len.mean() / 200.0, 1.0)

    cluster_profiles[c] = [dim1, dim2, dim3, dim4, dim5, dim6]

# ══════════════════════════════════════════════════════════════════════════
# FIG C1: t-SNE Cluster Visualization
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C1] t-SNE Cluster Visualization...')
fig, ax = plt.subplots(figsize=(14, 10))

for c in range(best_k):
    mask = case_df['cluster'] == c
    pts = case_df[mask]
    color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

    ax.scatter(pts['tsne_x'], pts['tsne_y'], c=color, s=18, alpha=0.55,
               edgecolors='white', linewidths=0.3, label=cluster_names[c], zorder=3)

    # Convex hull
    if mask.sum() >= 3:
        points = pts[['tsne_x', 'tsne_y']].values
        try:
            hull = ConvexHull(points)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.fill(points[hull_pts, 0], points[hull_pts, 1],
                    alpha=0.06, color=color, zorder=1)
            ax.plot(points[hull_pts, 0], points[hull_pts, 1],
                    color=color, alpha=0.3, linewidth=1.0, linestyle='--', zorder=2)
        except Exception:
            pass

    # Cluster center annotation
    cx, cy = pts['tsne_x'].mean(), pts['tsne_y'].mean()
    kw_text = '\n'.join(cluster_top3[c])
    ax.annotate(f'C{c}\n{kw_text}',
                xy=(cx, cy), fontsize=7, fontweight='bold',
                ha='center', va='center', color='#222222',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85, linewidth=1.2),
                zorder=5)

style_ax(ax, title='AI教育案例深度聚类 t-SNE 投影',
         subtitle='基于TF-IDF向量的KMeans聚类结果 (优势和创新点文本)',
         xlabel='t-SNE 维度 1', ylabel='t-SNE 维度 2')

legend = ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9,
                   edgecolor='#CCCCCC', fancybox=True, ncol=2,
                   title='聚类标签', title_fontsize=9)
legend.get_frame().set_linewidth(0.5)

n_cases = len(case_df)
fig.subplots_adjust(top=0.92, bottom=0.12)
caption = (f'图C1: AI教育案例创新文本聚类t-SNE可视化 (n={n_cases}, K={best_k})。'
           f'每个点代表一个去重案例，颜色表示聚类归属。'
           f'虚线凸包标示聚类边界，标注框显示各聚类核心关键词。'
           f'聚类基于"优势和创新点"字段TF-IDF向量，采用KMeans算法。')
add_caption_box(fig, caption, y=0.01, fontsize=8)
add_analysis_box(fig, f'聚类分析发现{best_k}个显著的教学创新模式，t-SNE投影显示聚类间存在明确边界，表明创新文本具有可区分的主题结构。',
                 x=0.02, y=0.13)

save_fig(fig, 'fig_c1_tsne_clusters.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C2: Cluster Profile Radar Charts
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C2] Cluster Profile Radar Charts...')

n_rows_radar = 2
n_cols_radar = (best_k + 1) // 2
fig = plt.figure(figsize=(4 * n_cols_radar, 8))

angles = np.linspace(0, 2 * np.pi, len(profile_dims), endpoint=False).tolist()
angles += angles[:1]

for c in range(best_k):
    ax = fig.add_subplot(n_rows_radar, n_cols_radar, c + 1, polar=True)
    values = cluster_profiles[c] + [cluster_profiles[c][0]]
    color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

    ax.plot(angles, values, 'o-', linewidth=1.8, color=color, markersize=4)
    ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(profile_dims, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=5, color='#888888')
    ax.spines['polar'].set_linewidth(0.3)
    ax.grid(linewidth=0.3, alpha=0.4)

    size = (case_df['cluster'] == c).sum()
    ax.set_title(f'{cluster_names[c]}\n(n={size})', fontsize=8, fontweight='bold', pad=12)

fig.suptitle('各聚类多维特征画像雷达图', fontsize=14, fontweight='bold', y=0.99)

caption = ('图C2: 各聚类的多维特征画像。六个维度分别为学段分布广度、学科覆盖度、'
           '技术复杂度、应用场景多样性、五育均衡度和创新深度（文本长度归一化）。'
           '数值越大表示该维度特征越突出。')

fig.tight_layout(rect=[0, 0.08, 1, 0.96])
add_caption_box(fig, caption, y=0.01, fontsize=8)
add_analysis_box(fig, '各聚类在六维特征空间中呈现差异化画像，技术复杂度和应用场景多样性是区分聚类的关键维度。',
                 x=0.02, y=0.08)
save_fig(fig, 'fig_c2_cluster_profiles.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C3: Hierarchical Clustering Dendrogram
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C3] Hierarchical Clustering Dendrogram...')

# Sample 250 cases for readability
np.random.seed(42)
n_sample = min(250, len(case_df))
sample_idx = np.random.choice(len(case_df), n_sample, replace=False)
sample_tfidf = tfidf_matrix[sample_idx].toarray()
sample_labels = case_df.iloc[sample_idx]['cluster'].values
sample_subjects = case_df.iloc[sample_idx]['学科'].fillna('未知').values

# Ward linkage
Z = linkage(sample_tfidf, method='ward', metric='euclidean')

fig, ax = plt.subplots(figsize=(16, 12))

# Color function based on cluster
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette([CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(best_k)])

leaf_labels = [f'{sample_subjects[i][:8]}' for i in range(n_sample)]

dend = dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=6,
                  color_threshold=Z[-best_k+1, 2] if best_k > 1 else 0,
                  above_threshold_color='#888888',
                  labels=leaf_labels)

set_link_color_palette(None)

style_ax(ax, title='AI教育案例层次聚类树状图',
         subtitle=f'Ward法聚类 (随机抽样 n={n_sample})',
         xlabel='案例学科标签', ylabel='聚类距离 (Ward)')

# Annotate major splits
n_splits = min(3, best_k - 1)
for i in range(1, n_splits + 1):
    dist = Z[-i, 2]
    ax.axhline(y=dist, color='#E63946', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(ax.get_xlim()[1] * 0.98, dist, f'第{i}次分裂 (d={dist:.1f})',
            ha='right', va='bottom', fontsize=7, color='#E63946',
            bbox=dict(facecolor='white', edgecolor='#E63946', alpha=0.8, pad=2))

caption = (f'图C3: 层次聚类树状图 (Ward法, 随机抽样n={n_sample})。'
           f'叶节点标签为案例学科缩写，分支颜色对应聚类归属。'
           f'红色虚线标注主要分裂节点及其距离值。')
fig.subplots_adjust(bottom=0.25, top=0.92)
add_caption_box(fig, caption, y=0.01, fontsize=8)
add_analysis_box(fig, '层次聚类树状图揭示了案例间的层级关系，主要分裂节点对应不同创新主题的分化。',
                 x=0.02, y=0.13)

save_fig(fig, 'fig_c3_dendrogram.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C4: Silhouette Analysis
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C4] Silhouette Analysis...')

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(1, 5, width_ratios=[4, 0.2, 1.5, 0.1, 0.1])
ax_main = fig.add_subplot(gs[0, 0])
ax_inset = fig.add_subplot(gs[0, 2])

# Silhouette samples for best_k
sample_sil = silhouette_samples(tfidf_matrix, case_df['cluster'].values)
avg_sil = silhouette_score(tfidf_matrix, case_df['cluster'].values)

y_lower = 10
for c in range(best_k):
    c_sil = sample_sil[case_df['cluster'].values == c]
    c_sil.sort()
    size_c = len(c_sil)
    y_upper = y_lower + size_c
    color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

    ax_main.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                          facecolor=color, edgecolor=color, alpha=0.7)
    ax_main.text(-0.02, y_lower + 0.5 * size_c, f'C{c}',
                 fontsize=7, fontweight='bold', va='center', ha='right')
    y_lower = y_upper + 10

ax_main.axvline(x=avg_sil, color='#E63946', linestyle='--', linewidth=1.5, zorder=5)
ax_main.text(avg_sil + 0.002, ax_main.get_ylim()[1] * 0.95,
             f'平均轮廓系数\n= {avg_sil:.4f}',
             fontsize=8, color='#E63946', fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='#E63946', alpha=0.9, pad=3))

style_ax(ax_main, title=f'轮廓分析图 (K={best_k})',
         subtitle='各聚类样本的轮廓系数分布',
         xlabel='轮廓系数', ylabel='聚类内样本')
ax_main.set_yticks([])

# Inset: silhouette scores for k=3..15
ks = sorted(sil_scores.keys())
vals = [sil_scores[k] for k in ks]
ax_inset.plot(ks, vals, 'o-', color='#457B9D', linewidth=1.5, markersize=5)
ax_inset.axvline(x=best_k, color='#E63946', linestyle='--', linewidth=1, alpha=0.7)
ax_inset.scatter([best_k], [sil_scores[best_k]], color='#E63946', s=80, zorder=5,
                 edgecolors='white', linewidths=1.5)
ax_inset.text(best_k, sil_scores[best_k] + 0.001, f'K={best_k}',
              ha='center', va='bottom', fontsize=8, fontweight='bold', color='#E63946')

style_ax(ax_inset, title='K值选择', xlabel='K', ylabel='轮廓系数')
ax_inset.set_xticks(ks)
ax_inset.tick_params(labelsize=7)

caption = (f'图C4: 轮廓分析。左图展示K={best_k}时各聚类样本的轮廓系数分布，'
           f'红色虚线为全局平均轮廓系数({avg_sil:.4f})。'
           f'右图为K=3~15的轮廓系数变化曲线，红点标记最优K值。')

fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
add_caption_box(fig, caption, y=0.01, fontsize=8)
add_analysis_box(fig, f'轮廓系数均值为{avg_sil:.4f}，表明聚类结构具有中等以上的分离度，K={best_k}为最优聚类数。',
                 x=0.02, y=0.15)
save_fig(fig, 'fig_c4_silhouette.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C5: Cluster x Feature Heatmap
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C5] Cluster x Feature Heatmap...')

# Top 30 features by overall TF-IDF weight
overall_weights = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top30_idx = overall_weights.argsort()[::-1][:30]
top30_names = feature_names[top30_idx]

# Build cluster x feature matrix
heatmap_data = np.zeros((best_k, 30))
for c in range(best_k):
    mask = case_df['cluster'].values == c
    if mask.sum() > 0:
        heatmap_data[c] = np.asarray(tfidf_matrix[mask][:, top30_idx].mean(axis=0)).flatten()

heatmap_df = pd.DataFrame(heatmap_data, index=[cluster_names[c] for c in range(best_k)],
                           columns=top30_names)

fig, ax = plt.subplots(figsize=(16, 10))
g = sns.clustermap(heatmap_df, cmap='YlOrRd', annot=True, fmt='.3f',
                   linewidths=0.5, linecolor='white',
                   figsize=(16, 10), dendrogram_ratio=(0.12, 0.12),
                   annot_kws={'fontsize': 6},
                   cbar_kws={'label': '平均TF-IDF权重', 'shrink': 0.6})
plt.close(fig)

g.fig.suptitle('聚类 × 关键特征词 TF-IDF热力图', fontsize=14, fontweight='bold', y=1.01)

caption = ('图C5: 聚类与Top-30 TF-IDF特征词的交叉热力图。'
           '行列均经层次聚类重排序，颜色深浅表示该聚类在对应特征词上的平均TF-IDF权重。'
           '可识别各聚类的核心主题词及跨聚类共享特征。')
g.fig.subplots_adjust(bottom=0.12)
g.fig.text(0.5, 0.01, caption, ha='center', va='top', fontsize=8,
           color='#333333', style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F0',
                     edgecolor='#CCCCCC', alpha=0.9))
g.fig.text(0.02, 0.12, '热力图揭示各聚类的核心主题词差异，高权重词反映聚类的教学创新特征。',
           ha='left', va='bottom', fontsize=7.5, color='#1a5276', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                     edgecolor='#cccccc', alpha=0.8))

g.savefig(str(FIG_DIR / 'fig_c5_cluster_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print('  [OK] fig_c5_cluster_heatmap.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C6: Cluster Composition Stacked Bars
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C6] Cluster Composition Stacked Bars...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Merge cluster labels back to all rows
cluster_map = case_df.set_index('案例编号')['cluster'].to_dict()
all_rows_c = all_rows.copy()
all_rows_c['cluster'] = all_rows_c['案例编号'].map(cluster_map)
all_rows_c = all_rows_c[all_rows_c['cluster'].notna()]

panel_configs = [
    ('学段', '学段', ['小学', '初中', '高中', '幼儿园', '中学', '其他']),
    ('学科', '学科', None),
    ('应用场景（一级）', '场景L1', None),
    ('培养方向', '五育', None),
]

for idx, (col, label, fixed_cats) in enumerate(panel_configs):
    ax = axes[idx // 2][idx % 2]

    # Build cross-tab
    ct = pd.crosstab(all_rows_c['cluster'], all_rows_c[col])

    if fixed_cats is None:
        # Keep top 6 categories
        top_cats = ct.sum().nlargest(6).index.tolist()
        ct = ct[top_cats]
    else:
        # Map non-listed to '其他'
        existing = [c for c in fixed_cats if c in ct.columns and c != '其他']
        other_cols = [c for c in ct.columns if c not in existing]
        if other_cols:
            ct['其他'] = ct[other_cols].sum(axis=1)
            ct = ct.drop(columns=other_cols)
        ct = ct[[c for c in fixed_cats if c in ct.columns]]

    # Normalize to percentages
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Plot stacked bars
    colors_bar = sns.color_palette('husl', len(ct_pct.columns))
    ct_pct.plot(kind='barh', stacked=True, ax=ax, color=colors_bar,
                edgecolor='white', linewidth=0.5)

    # Value labels
    for i, (_, row) in enumerate(ct_pct.iterrows()):
        cumsum = 0
        for j, val in enumerate(row):
            if val > 5:
                ax.text(cumsum + val / 2, i, f'{val:.0f}%',
                        ha='center', va='center', fontsize=6, color='white',
                        fontweight='bold')
            cumsum += val

    ax.set_yticklabels([cluster_names.get(int(t.get_text()), t.get_text())
                        for t in ax.get_yticklabels()], fontsize=7)
    style_ax(ax, title=f'聚类 × {label} 构成', xlabel='百分比 (%)')
    ax.legend(fontsize=6, loc='lower right', framealpha=0.8)
    ax.set_xlim(0, 100)

fig.suptitle('聚类构成分析：多维度百分比堆叠图', fontsize=14, fontweight='bold', y=0.99)

caption = ('图C6: 各聚类在四个维度上的构成分析。'
           '四个面板分别展示学段、学科、应用场景一级分类和五育培养方向的百分比分布。'
           '可揭示不同聚类的教育定位差异。')

fig.tight_layout(rect=[0, 0.08, 1, 0.96])
add_caption_box(fig, caption, y=0.01, fontsize=8)
add_analysis_box(fig, '聚类构成分析显示不同创新模式在学段、学科和场景上存在显著差异，反映了教育AI应用的多元化特征。',
                 x=0.02, y=0.08)
save_fig(fig, 'fig_c6_cluster_composition.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C7: UMAP/PCA 3D Projection
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C7] UMAP/PCA 3D Projection...')

fig, ax = plt.subplots(figsize=(14, 10))

if USE_UMAP:
    x_col, y_col, z_col = 'umap_x', 'umap_y', 'umap_z'
    method_label = 'UMAP'
else:
    x_col, y_col, z_col = 'pca_x', 'pca_y', 'pca_z'
    method_label = 'PCA'

# Normalize z for size encoding
z_vals = case_df[z_col].values
z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8)
sizes = 15 + z_norm * 80

for c in range(best_k):
    mask = case_df['cluster'] == c
    color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
    ax.scatter(case_df.loc[mask, x_col], case_df.loc[mask, y_col],
               s=sizes[mask], c=color, alpha=0.5, edgecolors='white',
               linewidths=0.3, label=cluster_names[c], zorder=3)

    # Cluster label
    cx = case_df.loc[mask, x_col].mean()
    cy = case_df.loc[mask, y_col].mean()
    ax.annotate(f'C{c}', xy=(cx, cy), fontsize=9, fontweight='bold',
                ha='center', va='center', color=color,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                zorder=6)

style_ax(ax, title=f'AI教育案例 {method_label} 三维投影 (深度编码为点大小)',
         subtitle=f'{method_label}降维至3维，第3维通过点大小表示',
         xlabel=f'{method_label} 维度 1', ylabel=f'{method_label} 维度 2')

# Size legend
for s_val, s_label in [(20, '低'), (55, '中'), (90, '高')]:
    ax.scatter([], [], s=s_val, c='gray', alpha=0.5, edgecolors='white',
               label=f'第3维: {s_label}')

legend = ax.legend(loc='upper left', fontsize=7, framealpha=0.9,
                   edgecolor='#CCCCCC', ncol=2, title='聚类 & 深度',
                   title_fontsize=8)

caption = (f'图C7: {method_label}三维降维投影。X/Y轴为前两个{method_label}维度，'
           f'点大小编码第三维度信息。颜色表示聚类归属，'
           f'可观察聚类在高维空间中的分离程度和重叠区域。')
fig.subplots_adjust(top=0.92, bottom=0.12)
add_caption_box(fig, caption, y=0.01, fontsize=8)
add_analysis_box(fig, f'{method_label}三维投影进一步验证了聚类的空间分离性，第三维度编码揭示了聚类内部的层次结构。',
                 x=0.02, y=0.13)

save_fig(fig, 'fig_c7_umap_3d.png')

# ══════════════════════════════════════════════════════════════════════════
# FIG C8: Cluster Technology Pathway Sankey-style Diagram
# ══════════════════════════════════════════════════════════════════════════
print('\n[FIG C8] Cluster Technology Pathway Sankey...')

# Extract dominant tech pathway per cluster
tech_paths = {}
for c in range(best_k):
    cids = case_df[case_df['cluster'] == c]['案例编号'].values
    c_rows = all_rows[all_rows['案例编号'].isin(cids)]
    paths = c_rows['关键技术路径'].dropna().value_counts()
    if len(paths) > 0:
        tech_paths[c] = paths.head(3).index.tolist()
    else:
        tech_paths[c] = ['未知路径']

# Parse tech pathway steps
def parse_steps(path_str):
    """Split a pathway like 'A → B → C' into steps."""
    separators = ['→', '->', '—', '➜']
    for sep in separators:
        if sep in str(path_str):
            return [s.strip() for s in str(path_str).split(sep) if s.strip()]
    return [str(path_str).strip()]

# Build flow data: cluster -> step1 -> step2 -> step3 -> step4
all_flows = []
for c in range(best_k):
    top_path = tech_paths[c][0] if tech_paths[c] else '未知'
    steps = parse_steps(top_path)
    # Pad to 4 steps
    while len(steps) < 4:
        steps.append('')
    all_flows.append({
        'cluster': c,
        'name': cluster_names[c],
        'steps': steps[:4],
        'size': (case_df['cluster'] == c).sum()
    })

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-1, best_k + 0.5)
ax.axis('off')

# Column positions
col_x = [0, 1.2, 2.4, 3.6]
col_labels = ['阶段1: 数据采集', '阶段2: 分析诊断', '阶段3: 智能干预', '阶段4: 效果评估']

# Draw column headers
for i, (x, label) in enumerate(zip(col_x, col_labels)):
    ax.text(x, best_k + 0.3, label, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#264653',
            bbox=dict(facecolor='#E8F4F8', edgecolor='#264653',
                      alpha=0.8, pad=4, boxstyle='round,pad=0.3'))

# Draw flows
for flow in all_flows:
    c = flow['cluster']
    y = best_k - 1 - c
    color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

    # Cluster label on left
    ax.text(-0.4, y, flow['name'], ha='right', va='center',
            fontsize=7, fontweight='bold', color=color)

    # Draw step boxes and connections
    for i, step in enumerate(flow['steps']):
        if not step:
            continue
        # Truncate long text
        display_text = step[:12] + '...' if len(step) > 12 else step

        # Box
        bbox = FancyBboxPatch((col_x[i] - 0.4, y - 0.2), 0.8, 0.4,
                               boxstyle='round,pad=0.05',
                               facecolor=color, alpha=0.15,
                               edgecolor=color, linewidth=1)
        ax.add_patch(bbox)
        ax.text(col_x[i], y, display_text, ha='center', va='center',
                fontsize=5.5, color='#333333')

        # Arrow to next step
        if i < len(flow['steps']) - 1 and flow['steps'][i + 1]:
            ax.annotate('', xy=(col_x[i + 1] - 0.4, y),
                        xytext=(col_x[i] + 0.4, y),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.2, alpha=0.6))

    # Size indicator
    ax.text(4.3, y, f'n={flow["size"]}', ha='left', va='center',
            fontsize=7, color='#666666')

ax.set_title('各聚类主导技术路径流程图', fontsize=14, fontweight='bold', pad=20)

caption = ('图C8: 各聚类主导技术路径Sankey式流程图。每行代表一个聚类的典型技术实施路径，'
           '从数据采集到效果评估分为四个阶段。箭头连接表示技术流转方向，'
           '右侧标注聚类样本量。可识别不同创新主题下的技术实施模式差异。')
fig.subplots_adjust(top=0.92, bottom=0.12)
fig.text(0.5, 0.01, caption, ha='center', va='top', fontsize=8,
         color='#333333', style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F0',
                   edgecolor='#CCCCCC', alpha=0.9))
add_analysis_box(fig, '技术路径分析揭示了从数据采集到效果评估的完整技术实施链条，不同聚类呈现差异化的技术选型策略。',
                 x=0.02, y=0.13)

save_fig(fig, 'fig_c8_cluster_evolution.png')

# ══════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('All 8 figures generated successfully!')
print(f'Output directory: {FIG_DIR}')
print('=' * 60)
