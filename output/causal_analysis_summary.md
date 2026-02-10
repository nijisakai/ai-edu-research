# AI Education Dataset: Comprehensive Causal & Statistical Analysis

## Overview

This report presents 10 multi-dimensional statistical analyses of 3,815 AI education product records.
All analyses handle missing data, report effect sizes alongside p-values, and use appropriate methods for the data types involved.

---
## 1. Correspondence Analysis (CA)

### 1a. Subject x Scenario
- **Total Inertia**: 0.7830
- **Dimension 1**: explains 50.6% of inertia
- **Dimension 2**: explains 38.2% of inertia
- **Contingency table**: 33 subjects x 6 scenarios

**Key Subject Coordinates (Dim 1):**

- 网络安全教育: -0.3492
- 语言: -0.3492
- 科学: -0.3463
  ...
- 心理健康教育: 2.1050
- 德育: 3.3232
- 美育: 4.9504

### 1b. Stage x Three-Empowerment
- **Total Inertia**: 0.0880
- **Dimension 1**: explains 65.8% of inertia
- **Dimension 2**: explains 20.9% of inertia

---
## 2. Multiple Correspondence Analysis (MCA)

- **Observations**: 3548
- **Components**: 3
- **Total Inertia**: 2.8000
- **Cumulative Dim 1**: 14.6%
- **Cumulative Dim 2**: 27.5%
- **Cumulative Dim 3**: 36.9%

### Dimension_1
**Negative pole** (low scores):
  - 产品技术代际__Gen5_多模态AI: -0.3047
  - iSTAR人机协同层级__HM2C(2): -0.2813
  - 智慧教育境界__第二境界_教学模式: -0.2611
**Positive pole** (high scores):
  - 智慧教育境界__第一境界_智慧环境: 0.4784
  - iSTAR人机协同层级__HUM(0): 7.2145
  - 三赋能分类__赋能学校: 7.5027

### Dimension_2
**Negative pole** (low scores):
  - 三赋能分类__赋能学校: -1.0877
  - iSTAR人机协同层级__HUM(0): -1.0172
  - iSTAR人机协同层级__HM2C(2): -1.0071
**Positive pole** (high scores):
  - 三赋能分类__赋能教师: 0.9654
  - 智慧教育境界__第三境界_制度变革: 1.2401
  - 三赋能分类__赋能评价: 2.0134

### Dimension_3
**Negative pole** (low scores):
  - 三赋能分类__赋能教师: -1.1265
  - 学段_clean__高中: -0.9591
  - 学段_clean__初中: -0.4996
**Positive pole** (high scores):
  - 产品技术代际__Gen3_AI辅助: 0.6457
  - 智慧教育境界__第三境界_制度变革: 2.2973
  - 三赋能分类__赋能评价: 3.1714

---
## 3. Ordinal Logistic Regression (DV: Innovation Depth)

- **N**: 3548
- **Pseudo R-squared**: 0.2012
- **AIC**: 7050.133419058166

### Proportional Odds Assumption
- LR statistic: -1435.34, df=48, p=0.0000
- Assumption holds: **False**

### Significant Predictors (p < 0.05)

| Predictor | Odds Ratio | 95% CI | p-value |
|-----------|-----------|--------|---------|
| 自研_num | 33.692 | [26.810, 42.339] | 0.0000 |
| iSTAR_num | 9.875 | [8.131, 11.992] | 0.0000 |
| 三赋能分类_赋能教师 | 0.093 | [0.053, 0.164] | 0.0000 |
| 三赋能分类_赋能评价 | 0.071 | [0.036, 0.140] | 0.0000 |
| tech_gen_num | 0.759 | [0.706, 0.816] | 0.0000 |
| 三赋能分类_赋能学生 | 0.140 | [0.078, 0.249] | 0.0000 |

---
## 4. Multinomial Logistic Regression (DV: iSTAR Level)

- **N**: 3548
- **Pseudo R-squared**: 0.2898

### Outcome: 0 (vs reference)

| Predictor | RRR | p-value | Sig |
|-----------|-----|---------|-----|
| tech_gen_num | 0.311 | 0.0000 | * |
| const | 34.752 | 0.0000 | * |
| 学科_grouped_未提及 | 10.666 | 0.0000 | * |
| 学科_grouped_物理 | 6.566 | 0.0000 | * |
| 学科_grouped_其他 | 2.662 | 0.0000 | * |
| D1_深度学习 | 0.624 | 0.0001 | * |
| 学科_grouped_语文 | 2.623 | 0.0001 | * |
| D3_循证教学 | 1.373 | 0.0014 | * |
| 学科_grouped_数学 | 2.126 | 0.0029 | * |
| 学科_grouped_美术 | 2.224 | 0.0043 | * |
| 学段_clean_幼儿园 | 0.715 | 0.0303 | * |

### Outcome: 1 (vs reference)

| Predictor | RRR | p-value | Sig |
|-----------|-----|---------|-----|
| tech_gen_num | 0.247 | 0.0000 | * |
| D1_深度学习 | 0.136 | 0.0000 | * |
| D3_循证教学 | 0.127 | 0.0000 | * |
| 学段_clean_高中 | 3.831 | 0.0224 | * |
| D2_绿色鲁棒 | 3.921 | 0.0440 | * |

---
## 5. Structural Equation Modeling (Path Analysis)

- **N**: 3815
- **Full Model R-squared**: 0.1072

### Path Coefficients

| Path | Coefficient | p-value |
|------|------------|---------|
| DevLevel_to_TechGen | -0.0386 | 0.0572 |
| TechGen_to_D4 | 0.0125 | 0.0043 |
| TechGen_to_D3 | -0.0473 | 0.0000 |
| D4_to_Innovation | 0.6547 | 0.0000 |
| D3_to_Innovation | 0.3966 | 0.0000 |
| TechGen_to_Innovation_direct | 0.0517 | 0.0000 |
| TechGen_to_Innovation_total | 0.0411 | 0.0010 |

### Decomposition of Effects (TechGeneration -> InnovationDepth)
- **Total effect**: 0.0411
- **Direct effect**: 0.0517
- **Indirect via D4 (Human-Machine Trust)**: 0.0082
- **Indirect via D3 (Evidence-Based Teaching)**: -0.0188
- **Proportion mediated**: -25.6%

### Mediation Tests
- D4 Sobel test: z=2.810, p=0.0050, significant=True
- D3 Sobel test: z=-6.051, p=0.0000, significant=True
- D4 Bootstrap 95% CI: (0.0028145599987198597, 0.014297416300916101), excludes zero: True
- D3 Bootstrap 95% CI: (-0.025027289667089454, -0.013094403724090575), excludes zero: True

---
## 6. Geographic Inequality Decomposition

### innovation_depth
- **Overall Gini**: 0.1966
- **Theil (province)**: total=0.0700, between=1.9%, within=98.1%
- **Theil (region)**: total=0.0700, between=0.3%, within=99.7%

**Top provinces by Gini (most unequal):**
  - 湖南省: 0.2509
  - 辽宁省: 0.2395
  - 河北省: 0.2269
  - 山西省: 0.2160
  - 吉林省: 0.2142

### iSTAR_level
- **Overall Gini**: 0.1903
- **Theil (province)**: total=0.0253, between=1.0%, within=99.0%
- **Theil (region)**: total=0.0253, between=0.3%, within=99.7%

**Top provinces by Gini (most unequal):**
  - 江西省: 0.2227
  - 江苏省: 0.2013
  - 山西省: 0.1994
  - 重庆市: 0.1977
  - 新疆维吾尔自治区: 0.1952

### tech_generation
- **Overall Gini**: 0.1842
- **Theil (province)**: total=0.0580, between=0.9%, within=99.1%
- **Theil (region)**: total=0.0580, between=0.5%, within=99.5%

**Top provinces by Gini (most unequal):**
  - 内蒙古自治区: 0.2172
  - 广东省: 0.2041
  - 上海市: 0.2031
  - 江苏省: 0.1980
  - 福建省: 0.1898

---
## 7. Cluster Profiling (K=10)

- **Observations**: 3815
- **Cluster sizes**: {'0': 271, '1': 528, '2': 542, '3': 865, '4': 245, '5': 54, '6': 393, '7': 321, '8': 373, '9': 223}

### ANOVA Results (Continuous Variables)

| Variable | F-statistic | p-value | Eta-squared | Effect Size |
|----------|------------|---------|-------------|-------------|
| 创新深度评分 | 352.68 | 0.00e+00 | 0.4548 | large |
| D1_深度学习 | 1198.08 | 0.00e+00 | 0.7392 | large |
| D3_循证教学 | 1482.11 | 0.00e+00 | 0.7781 | large |
| D4_人机互信 | 38359.08 | 0.00e+00 | 0.9891 | large |
| tech_gen_num | 413.11 | 0.00e+00 | 0.4942 | large |

### Chi-Square Results (Categorical Variables)

| Variable | Chi2 | p-value | Cramer's V | Effect Size |
|----------|------|---------|-----------|-------------|
| 学段_clean | 417.93 | 1.02e-43 | 0.1103 | small |
| 应用场景（一级） | 1446.53 | 4.38e-267 | 0.2514 | small |
| 产品技术代际 | 2360.28 | 0.00e+00 | 0.3933 | medium |
| 三赋能分类 | 1322.97 | 5.65e-262 | 0.3400 | medium |

---
## 8. Association Rule Mining (Apriori)

- **Transactions**: 3548
- **Frequent itemsets**: 149
- **Rules found**: 443
- **Lift range**: [0.66, 6.12], mean=1.71

### Top 10 Rules by Lift

| # | Antecedents | Consequents | Support | Confidence | Lift |
|---|------------|------------|---------|-----------|------|
| 1 | 场景_助教 | 学科_其他学科, 赋能_赋能教师, 协同_HMC(1) | 0.051 | 0.314 | 6.12 |
| 2 | 学科_其他学科, 赋能_赋能教师, 协同_HMC(1) | 场景_助教 | 0.051 | 1.000 | 6.12 |
| 3 | 学科_其他学科, 赋能_赋能教师 | 场景_助教 | 0.052 | 1.000 | 6.12 |
| 4 | 场景_助教 | 学科_其他学科, 赋能_赋能教师 | 0.052 | 0.319 | 6.12 |
| 5 | 场景_助教, 协同_HMC(1) | 学段_小学, 赋能_赋能教师 | 0.083 | 0.510 | 6.10 |
| 6 | 学段_小学, 赋能_赋能教师 | 场景_助教, 协同_HMC(1) | 0.083 | 0.987 | 6.10 |
| 7 | 场景_助教, 协同_HMC(1) | 学科_其他学科, 赋能_赋能教师 | 0.051 | 0.317 | 6.08 |
| 8 | 学科_其他学科, 赋能_赋能教师 | 场景_助教, 协同_HMC(1) | 0.051 | 0.984 | 6.08 |
| 9 | 学段_小学, 赋能_赋能教师 | 场景_助教 | 0.083 | 0.987 | 6.03 |
| 10 | 学段_小学, 协同_HMC(1), 赋能_赋能教师 | 场景_助教 | 0.083 | 0.987 | 6.03 |

---
## 9. Interaction Effects (Factorial ANOVA)

### stage_x_techgen_on_innovation
- **N**: 3548, R-squared: 0.0255

| Source | SS | df | F | p-value | Eta-sq | Effect |
|--------|----|----|---|---------|--------|--------|
| C(Stage) | 4.50 | 3 | 2.11 | 0.0972 | 0.0017 | small |
| C(TechGen) | 49.48 | 4 | 17.37 | 0.0000 | 0.0192 | small |
| C(Stage):C(TechGen) | 12.37 | 12 | 1.45 | 0.1368 | 0.0048 | small |

### stage_x_empowerment_on_istar
- **N**: 3548, R-squared: 0.2857

| Source | SS | df | F | p-value | Eta-sq | Effect |
|--------|----|----|---|---------|--------|--------|
| C(Stage) | 5.47 | 3 | 9.17 | 0.0001 | 0.0052 | small |
| C(Empower) | 335.14 | 3 | 562.08 | 0.0000 | 0.3211 | large |
| C(Stage):C(Empower) | 0.91 | 9 | 0.51 | 0.8524 | 0.0009 | small |

### region_x_techgen_on_innovation
- **N**: 3698, R-squared: 0.0227

| Source | SS | df | F | p-value | Eta-sq | Effect |
|--------|----|----|---|---------|--------|--------|
| C(Region) | 6.70 | 6 | 1.56 | 0.1555 | 0.0025 | small |
| C(TechGen) | 3.89 | 4 | 1.36 | 0.2471 | 0.0014 | small |
| C(Region):C(TechGen) | 78.45 | 24 | 4.56 | 0.0328 | 0.0289 | small |

---
## 10. Predictive Modeling (Random Forest + SHAP)

- **N**: 3511, Features: 18
- **Model**: RandomForestClassifier(n_estimators=300, max_depth=12, balanced)
- **5-fold CV Accuracy**: 0.6633 (+/- 0.0106)
- **5-fold CV F1 (macro)**: 0.6944 (+/- 0.0259)
- **5-fold CV F1 (weighted)**: 0.6649 (+/- 0.0109)
- **Cohen's Kappa (full data)**: 0.7670

### Feature Importance (Top 15)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | 自研_num | 0.2214 |
| 2 | D3_循证教学 | 0.1394 |
| 3 | iSTAR_num | 0.0778 |
| 4 | iSTAR人机协同层级 | 0.0721 |
| 5 | D1_深度学习 | 0.0684 |
| 6 | D4_人机互信 | 0.0542 |
| 7 | 产品分类 | 0.0494 |
| 8 | 学段_clean | 0.0478 |
| 9 | 产品形态 | 0.0443 |
| 10 | 区域 | 0.0432 |
| 11 | 产品技术代际 | 0.0328 |
| 12 | tech_gen_num | 0.0312 |
| 13 | 应用场景（一级） | 0.0268 |
| 14 | dev_level | 0.0241 |
| 15 | smart_edu_num | 0.0202 |

### SHAP Values (Top 15)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | 自研_num | 0.0869 |
| 2 | D3_循证教学 | 0.0644 |
| 3 | iSTAR_num | 0.0454 |
| 4 | iSTAR人机协同层级 | 0.0414 |
| 5 | D1_深度学习 | 0.0337 |
| 6 | D4_人机互信 | 0.0234 |
| 7 | 产品分类 | 0.0151 |
| 8 | 应用场景（一级） | 0.0126 |
| 9 | 三赋能分类 | 0.0097 |
| 10 | 产品技术代际 | 0.0096 |
| 11 | 学段_clean | 0.0092 |
| 12 | tech_gen_num | 0.0088 |
| 13 | 区域 | 0.0079 |
| 14 | 产品形态 | 0.0070 |
| 15 | dev_level | 0.0059 |

---
## Key Findings Summary

1. **Correspondence Analysis**: The Subject x Scenario CA reveals that the first two dimensions capture 88.8% of total inertia (Dim1=50.6%, Dim2=38.2%). Subjects like DE (moral education) and aesthetic education cluster far from the mainstream "intelligent tutoring" pole, while STEM subjects cluster near "助学" (learning assistance). The Stage x Empowerment CA shows strong structure (total inertia=0.088), with kindergarten distinctly separated from secondary stages.

2. **MCA** (N=3,548): Three latent dimensions explain 36.9% of total inertia. Dimension 1 contrasts rare institutional empowerment (school-level, HUM) against the mainstream student-empowerment + advanced collaboration pole. Dimension 2 separates teacher empowerment and institutional reform from student-facing patterns. Dimension 3 captures the assessment/evaluation vs. teaching orientation.

3. **Ordinal Logistic Regression** (N=3,548, pseudo R2=0.201): Self-developed products have the strongest effect (OR=33.7, p<0.001), followed by iSTAR level (OR=9.9, p<0.001). Three-empowerment categories of "赋能教师" (OR=0.09) and "赋能评价" (OR=0.07) are associated with lower innovation depth relative to school empowerment. The proportional odds assumption is violated (p<0.001), suggesting the effects differ across innovation score thresholds.

4. **Multinomial Logistic Regression** (pseudo R2=0.290): Predicting iSTAR level, tech generation is the strongest differentiator (RRR=0.25 for HMC vs reference). D1 (deep learning) and D3 (evidence-based teaching) have opposing effects: D1=0.14 (strongly reduces odds of HMC) while D3=0.13 (reduces odds of HMC), indicating advanced collaboration requires both deep learning AND evidence-based approaches.

5. **Path Analysis** (N=3,815, R2=0.107): TechGeneration has a significant total effect on InnovationDepth (beta=0.041, p=0.001). The D4 (Human-Machine Trust) pathway mediates positively (indirect=0.008, Sobel z=2.81, p=0.005, bootstrap CI excludes zero). Counterintuitively, the D3 (Evidence-Based Teaching) pathway mediates negatively (indirect=-0.019, Sobel z=-6.05, p<0.001), because higher tech generations are actually associated with *lower* D3 scores (beta=-0.047), suggesting newer technologies have not yet integrated evidence-based practices.

6. **Geographic Inequality**: Inequality is predominantly within-province rather than between-province. For innovation depth, 98.1% of Theil index variance is within-province (Gini=0.197). The most unequal provinces are Hunan (Gini=0.251), Liaoning (0.240), and Hebei (0.227). Between-region inequality accounts for only 0.3-0.5%, suggesting the digital divide in AI education is local rather than regional.

7. **Cluster Profiling** (K=10): All continuous ANOVA tests yield large effect sizes. D4 (Human-Machine Trust) has the highest discrimination (eta2=0.989, F=38,359), followed by D3 (eta2=0.778) and D1 (eta2=0.739). Innovation depth shows eta2=0.455. Categorical chi-square tests show tech generation (Cramer's V=0.393, medium) and three-empowerment (V=0.340, medium) best differentiate clusters, while educational stage has weaker discrimination (V=0.110, small).

8. **Association Rules** (443 rules, lift range [0.66, 6.12]): The strongest rules (lift~6.1) consistently link "助教" (teaching assistance) scenarios with "赋能教师" (teacher empowerment) and HMC(1) collaboration level. Elementary school + teacher empowerment is nearly deterministic for the "助教" scenario (confidence=0.987). This reveals a tight coupling between teacher-oriented products and medium-collaboration designs.

9. **Interaction Effects**: The Stage x Empowerment interaction on iSTAR level is the most substantive model (R2=0.286), where empowerment type alone explains 32.1% of variance (eta2=0.321, large effect). The Region x TechGeneration interaction is statistically significant (p=0.033) but small (eta2=0.029), suggesting regional differences in how technology generations map to innovation. The Stage x TechGeneration interaction is not significant (p=0.137).

10. **Predictive Modeling** (Random Forest, 5-fold CV accuracy=66.3%, Cohen's kappa=0.767): Self-development status is the top predictor (importance=0.221, SHAP=0.087), followed by D3 evidence-based teaching (importance=0.139, SHAP=0.064) and iSTAR level (importance=0.078, SHAP=0.045). The SHAP and Gini importance rankings are highly consistent, confirming that process-oriented features (D-scores, collaboration level) outweigh structural features (region, tech generation) in predicting innovation depth.

---
*Analysis generated by causal_analysis.py pipeline*