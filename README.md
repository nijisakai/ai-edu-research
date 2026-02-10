# AI赋能基础教育的实践图景与产业生态研究

## AI-Empowered K-12 Education: Practice Landscape and Industry Ecosystem

> 基于1690个典型案例的混合方法研究 | A Mixed-Methods Study Based on 1,690 Representative Cases

---

## 研究概述

本项目对全国基础教育教师AI教学案例征集活动所获取的**1,690个典型案例**（涵盖**3,815条工具/产品记录**）进行了系统性的多维分析，从**教育、技术、产业**三个维度全面解构了当前AI赋能基础教育的实践样态。

### 核心发现

| 维度 | 关键发现 |
|------|----------|
| **数据规模** | 1,690案例 · 3,815工具记录 · 30省份 · 1,830种工具 · 1,726家企业 |
| **教育格局** | 小学主导(52.7%) · 助学为主(73.5%) · 智育偏重(72.6%) · 五育失衡 |
| **技术栈** | 大模型驱动 · 豆包(11.4%)+DeepSeek(8.0%)领跑 · 软件平台化(73.6%) |
| **产业生态** | 高度碎片化 · Top5仅占28.2% · 自主研发率14.4% · 规模化应用初期 |
| **区域差异** | 浙江领跑(25.6%) · 东强西弱 · 数字鸿沟收窄中 |

---

## 项目结构

```
ai-edu-research/
├── src/
│   ├── core_analysis.py          # 核心统计分析
│   ├── nlp_analysis.py           # NLP语义分析（TF-IDF/LDA/聚类）
│   └── visualizations.py         # 18张出版级可视化
├── output/
│   ├── figures/                   # 18张高质量图表（PNG+PDF）
│   │   ├── fig01_province_heatmap.png
│   │   ├── fig02_school_stage_donut.png
│   │   ├── fig03_top20_tools_bar.png
│   │   ├── fig04_product_ecosystem_treemap.png
│   │   ├── fig05_scenario_sunburst.png
│   │   ├── fig06_subject_distribution.png
│   │   ├── fig07_cultivation_radar.png
│   │   ├── fig08_tech_wordcloud.png
│   │   ├── fig09_tech_cooccurrence_network.png
│   │   ├── fig10_stage_subject_heatmap.png
│   │   ├── fig11_industry_maturity.png
│   │   ├── fig12_self_developed_ratio.png
│   │   ├── fig13_company_market_concentration.png
│   │   ├── fig14_lda_topics.png
│   │   ├── fig15_tech_pathway_sankey.png
│   │   ├── fig16_scenario_stage_bubble.png
│   │   ├── fig17_innovation_cluster.png
│   │   └── fig18_comprehensive_dashboard.png
│   ├── paper/                     # 学术论文
│   │   └── 论文_AI赋能基础教育的实践图景与产业生态.md
│   ├── report/                    # 研究报告
│   │   └── 研究报告_AI赋能基础教育实践图景与产业生态分析.md
│   ├── *.json                     # 分析结果数据
│   ├── huang_ronghuai_research.md # 黄荣怀教授理论研究
│   └── industry_research.md       # 行业调研资料
├── data/                          # 数据目录
├── requirements.txt               # Python依赖
└── README.md                      # 本文件
```

---

## 研究方法

本研究采用**混合方法**（Mixed Methods）研究设计：

### 1. 描述性统计分析
- 学段、学科、地域、工具使用等结构化字段的频次统计与交叉分析

### 2. NLP文本挖掘
- **jieba分词** + 自定义教育/技术词典
- **TF-IDF关键词提取**：识别各文本字段的核心概念
- **LDA主题建模**：发现7个潜在教学创新主题
- **词共现网络**：构建技术要素语义关联图谱

### 3. 聚类分析
- **KMeans聚类**（k=10）：基于TF-IDF向量的案例分群
- **PCA降维可视化**：展示案例在语义空间中的分布

### 4. 技术路径挖掘
- 解析1,537条独特技术路径链
- 识别主导模式："学习行为采集→学情诊断→个性化反馈→学习改进"

### 5. 产业生态分析
- 市场集中度（HHI指数、CR5/CR10）
- 产品形态与分类矩阵
- 自主研发率评估

---

## 可视化图表

| 图号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| Fig.01 | 省域分布热力图 | 渐变条形图 | 30省案例分布 |
| Fig.02 | 学段分布环形图 | 环形图 | 四大学段占比 |
| Fig.03 | Top20 AI工具 | 水平条形图 | 头部工具市场份额 |
| Fig.04 | 产品生态树图 | Treemap | 产品分类层级 |
| Fig.05 | 场景旭日图 | 嵌套饼图 | 场景L1→L2层级 |
| Fig.06 | 学科分布图 | 极坐标图 | 15+学科渗透率 |
| Fig.07 | 五育雷达图 | 雷达图 | 各学段五育融合 |
| Fig.08 | 技术词云 | 词云 | 创新关键词 |
| Fig.09 | 技术共现网络 | 网络图 | 技术要素关联 |
| Fig.10 | 学段×学科热力图 | 热力图 | 交叉分布矩阵 |
| Fig.11 | 产业成熟度 | 条形图 | 发展阶段分布 |
| Fig.12 | 自研比例 | 对比图 | 自研vs第三方 |
| Fig.13 | 市场集中度 | 洛伦兹曲线 | 企业市场份额 |
| Fig.14 | LDA主题 | 主题词图 | 7大创新主题 |
| Fig.15 | 技术路径桑基图 | 桑基图 | 技术实施路径 |
| Fig.16 | 场景×学段气泡图 | 气泡图 | 多维交叉分析 |
| Fig.17 | 创新聚类散点图 | PCA散点图 | 10类案例分群 |
| Fig.18 | 综合仪表盘 | 组合图 | 核心指标总览 |

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行核心分析
python src/core_analysis.py

# 运行NLP语义分析
python src/nlp_analysis.py

# 生成全部可视化
python src/visualizations.py
```

---

## 理论框架

本研究基于以下理论框架：

- **黄荣怀智慧教育框架**（Smart Education Framework）
- **TPACK模型**（Technological Pedagogical Content Knowledge）
- **SAMR模型**（Substitution-Augmentation-Modification-Redefinition）
- **UNESCO AI教育指南**
- **教育-技术-产业三维分析框架**（本研究构建）

---

## 引用

如需引用本研究，请使用：

```
人工智能赋能基础教育的实践图景与产业生态——基于1690个典型案例的混合方法研究, 2026.
```

---

## License

MIT License
