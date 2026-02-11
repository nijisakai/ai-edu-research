# AI赋能基础教育的实践图景与产业生态研究

## AI-Empowered K-12 Education: Practice Landscape and Industry Ecosystem

> 基于1690个典型案例的混合方法研究 | A Mixed-Methods Study Based on 1,690 Representative Cases

**[交互式仪表板 Live Dashboard](https://nijisakai.github.io/ai-edu-research/)** | **[数据API](https://nijisakai.github.io/ai-edu-research/api.html)**

---

## 研究概述

本项目对全国基础教育教师AI教学案例征集活动所获取的**1,690个典型案例**（涵盖**3,815条工具/产品记录**）进行了系统性的多维分析，从**教育、技术、产业**三个维度全面解构了当前AI赋能基础教育的实践样态。

### 核心发现

| 维度 | 关键发现 |
|------|----------|
| **数据规模** | 1,690案例 · 3,815工具记录 · 28省份 · 1,830种工具 · 1,726家企业 |
| **教育格局** | 小学主导(52.7%) · 助学为主(73.5%) · 智育偏重(72.6%) · 五育失衡 |
| **技术栈** | 大模型驱动 · 豆包(11.4%)+DeepSeek(8.0%)领跑 · 软件平台化(73.6%) |
| **产业生态** | 高度碎片化 · Top5仅占28.2% · 自主研发率14.4% · 规模化应用初期 |
| **区域差异** | 浙江领跑(25.6%) · 东强西弱 · 98.1%不平等来自省内而非省际 |

---

## 在线资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 交互式仪表板 | [nijisakai.github.io/ai-edu-research](https://nijisakai.github.io/ai-edu-research/) | 8个交互图表：省域分布、工具生态、场景分析、理论框架、语义聚类、创新深度、五育雷达、案例浏览 |
| 数据API | [api.html](https://nijisakai.github.io/ai-edu-research/api.html) | 28个JSON数据集的RESTful接口，支持跨域访问 |

---

## 项目结构

```
ai-edu-research/
├── src/                              # Python分析模块（16个，15.5K行代码）
│   ├── core_analysis.py              # 核心统计分析
│   ├── nlp_analysis.py               # NLP语义分析（TF-IDF/LDA/聚类）
│   ├── causal_analysis.py            # 因果与统计分析（10种方法）
│   ├── deep_clustering.py            # 深度聚类分析
│   ├── insight_mining.py             # 深度洞察挖掘
│   ├── visualizations.py             # 核心可视化（18张）
│   ├── premium_viz.py                # 出版级可视化
│   ├── framework_viz.py              # 理论框架可视化
│   ├── viz_part1.py / viz_part2.py   # 框架可视化扩展
│   ├── advanced_stats.py             # 高级统计可视化
│   ├── relabel_framework.py          # 理论框架标注管线
│   ├── rebuild_all.py                # 全量构建管线（MD→HTML→PDF）
│   └── render_paper_pdf.py / render_report_pdf.py / render_pdfs.py
│
├── data/                             # 原始数据
│   ├── 教育产品统计_V5.csv            # 主数据集（6.4MB, 3815行×31列）
│   └── all_papers.json               # 参考文献
│
├── output/                           # 分析输出
│   ├── figures/                      # 87张高质量图表（PNG 300DPI + PDF）
│   ├── paper/                        # 学术论文（MD/HTML/PDF）
│   ├── report/                       # 研究报告（MD/HTML/PDF）
│   ├── *.json                        # 28个分析结果数据文件
│   ├── 案例深度分析摘要.md             # 46案例深度分析（中文）
│   ├── 因果与统计分析报告.md           # 10项因果统计分析（中文）
│   ├── 深度洞察挖掘报告.md             # 深度洞察报告（中文）
│   ├── AI基础教育产业研究简报.md        # 产业研究简报（中文）
│   └── huang_ronghuai_research.md    # 黄荣怀教授理论研究
│
├── docs/                             # 交互式仪表板（GitHub Pages）
│   ├── index.html                    # 主仪表板（8个Plotly.js图表）
│   ├── api.html                      # 数据API文档
│   ├── css/dashboard.css
│   ├── js/app.js                     # 应用控制器
│   ├── js/charts/                    # 8个图表模块
│   └── data/                         # JSON数据副本
│
├── quarto-site/                      # Quarto学术文档站点
│   ├── _quarto.yml                   # 站点配置
│   ├── index.qmd                     # 首页
│   ├── methodology.qmd              # 研究方法
│   ├── findings/                     # 核心发现（教育/技术/产业）
│   └── case-studies/                 # 案例研究
│
├── .github/workflows/                # GitHub Actions
│   ├── deploy-pages.yml              # 自动部署仪表板到GitHub Pages
│   └── run-analysis.yml              # 手动触发Python分析管线
│
├── 论文审读报告.md                     # 论文学术审读
├── 报告审读报告.md                     # 报告质量审读
├── 技术说明与图表解读指南.md            # 技术原理与87张图表详解
├── requirements.txt                  # Python依赖
└── README.md
```

---

## 研究方法

本研究采用**混合方法**（Mixed Methods）研究设计，包含6大分析模块：

### 1. 描述性统计分析
- 省份、学段、学科、工具、场景等多维度分布与交叉分析

### 2. NLP文本挖掘
- **jieba分词** + 自定义教育/技术词典
- **TF-IDF关键词提取**：核心概念识别
- **LDA主题建模**：7个潜在教学创新主题
- **KMeans语义聚类**（k=10）：案例分群与画像

### 3. 因果与统计分析（10种方法）
| 方法 | 核心发现 |
|------|---------|
| 对应分析(CA) | 学科×场景前两维解释88.8%惯性 |
| 多重对应分析(MCA) | 三维解释36.9%惯性 |
| 序数Logistic回归 | 自研产品OR=33.7，iSTAR层级OR=9.9 |
| 多项Logistic回归 | 技术代际为最强区分变量 |
| 路径分析(SEM) | D4正向中介，D3反向中介 |
| 地理不平等分解 | 98.1%不平等来自省内 |
| 聚类画像(K=10) | D4区分力最高(η²=0.989) |
| 关联规则(Apriori) | 助教→赋能教师+HMC(1)，lift=6.12 |
| 交互效应(ANOVA) | 赋能类型解释32.1%方差 |
| 随机森林+SHAP | 自研状态为最重要特征(0.221) |

### 4. 深度案例分析
- 46个代表性案例的定性编码与理论框架标注

### 5. 深度洞察挖掘
- 工具生态分析、数字鸿沟诊断、创新指标评估、教师创造性模式

### 6. 产业生态分析
- 市场集中度（HHI/CR5/CR10）、产品形态矩阵、自主研发率

---

## 理论框架

| 框架 | 维度 | 应用 |
|------|------|------|
| **iSTAR人机协同模型** | HUM → HMC(1) → HM2C(2) → HMnC(3) | 案例人机协同层级评估 |
| **智慧教育三重境界** | 智慧环境 → 教学模式 → 制度变革 | 教育智慧化阶段判定 |
| **数字教学法四维度** | 深度学习·绿色鲁棒·循证教学·人机互信 | 教学法创新评估 |
| **三赋能分类** | 赋能学生·赋能教师·赋能评价·赋能学校 | 技术赋能方向分类 |
| **六大技术路径** | 内容生成·智能评测·数据驱动·沉浸体验·智能硬件·平台生态 | 技术实施路径分析 |

---

## 可视化图表

87张高质量图表分为7个系列：

| 系列 | 数量 | 内容 |
|------|------|------|
| A系列 | 5张 | 地理与结构分析（省域地图、学段分布、学科分布、场景树图、五育雷达） |
| B系列 | 4张 | 工具与技术（Top20工具、技术桑基图、模型图景、共现网络） |
| C系列 | 8张 | 深度聚类（UMAP、t-SNE、聚类画像、谱系图、轮廓分析） |
| D系列 | 2张 | 产业生态（洛伦兹曲线、生态树图） |
| E系列 | 6张 | 创新与框架（三赋能×iSTAR、创新山脊图、三重境界、数字教学法雷达） |
| F系列 | 8张 | 框架分析（框架热力图、技术代际流图、桑基图、省域框架） |
| S系列 | 10张 | 补充统计（对应分析、卡方马赛克、相关矩阵、回归图、冲积图） |

---

## 快速开始

### 本地运行分析

```bash
pip install -r requirements.txt

python src/core_analysis.py       # 核心统计
python src/nlp_analysis.py        # NLP语义分析
python src/causal_analysis.py     # 因果统计分析
python src/visualizations.py      # 生成可视化
python src/rebuild_all.py         # 全量构建（MD→HTML→PDF）
```

### 本地预览仪表板

```bash
cd docs && python -m http.server 8080
# 访问 http://localhost:8080
```

### Quarto学术站点（需安装Quarto）

```bash
cd quarto-site && quarto preview
```

---

## CI/CD

| 工作流 | 触发条件 | 功能 |
|--------|---------|------|
| `deploy-pages.yml` | push到main | 自动同步JSON数据并部署仪表板到GitHub Pages |
| `run-analysis.yml` | 手动触发 | 可选运行core/nlp/viz/causal/all分析模块 |

---

## 引用

如需引用本研究，请使用：

```
人工智能赋能基础教育的实践图景与产业生态——基于1690个典型案例的混合方法研究, 2026.
```

---

## License

MIT License
