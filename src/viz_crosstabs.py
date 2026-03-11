#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate visualizations for Section 2.2.4
1. Region x Product
2. Stage x Product
3. Subject x Product
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "new_reviews" / "V5.xlsx"
OUTPUT_DIR = BASE_DIR / "output" / "figures"

# Style settings
plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def normalize_stage(s):
    if pd.isna(s): return "未知"
    s = str(s).strip()
    mapping = {"小学": "小学", "初中": "初中", "高中": "高中", "中学": "中学", "幼儿园": "幼儿园", "学前": "幼儿园", "职高": "职业高中", "中职": "职业高中", "职业高中": "职业高中"}
    for k, v in mapping.items():
        if k in s: return v
    return s

def process():
    df = pd.read_excel(CSV_PATH, engine='calamine')
    
    # Process Region
    province_region = {
        '北京市': '东部', '天津市': '东部', '河北省': '东部', '上海市': '东部', '江苏省': '东部', '浙江省': '东部', '福建省': '东部', '山东省': '东部', '广东省': '东部', '海南省': '东部', '辽宁省': '东部',
        '山西省': '中部', '吉林省': '中部', '黑龙江省': '中部', '安徽省': '中部', '江西省': '中部', '河南省': '中部', '湖北省': '中部', '湖南省': '中部',
        '内蒙古自治区': '西部', '广西壮族自治区': '西部', '重庆市': '西部', '四川省': '西部', '贵州省': '西部', '云南省': '西部', '西藏自治区': '西部', '陕西省': '西部', '甘肃省': '西部', '青海省': '西部', '宁夏回族自治区': '西部', '新疆维吾尔自治区': '西部',
    }
    df['区域'] = df['省份'].fillna('').map(province_region).fillna('未知')
    df['学段_标准化'] = df['学段'].apply(normalize_stage)
    
    df['tools_list'] = df['工具标准名'].fillna('未知')
    df_tools = df[df['tools_list'] != '未知'].copy()
    
    # Top 8 products overall to filter the columns
    top_n_tools = df_tools['tools_list'].value_counts().head(8).index.tolist()
    # Filter the dataset slightly so we only plot for top tools
    df_plot = df_tools[df_tools['tools_list'].isin(top_n_tools)].copy()

    # 1. Region x Product Heatmap
    region_tool = pd.crosstab(df_plot['区域'], df_plot['tools_list'])
    region_tool = region_tool.loc[['东部', '中部', '西部'], top_n_tools] # reorder
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(region_tool, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('图5 典型区域与头部AI产品联动应用热力图', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('AI产品', fontsize=12)
    plt.ylabel('区域', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_a06_region_product.png')
    plt.close()

    # 2. Stage x Product Heatmap
    stage_tool = pd.crosstab(df_plot['学段_标准化'], df_plot['tools_list'])
    # Only keep main stages
    stages = ['幼儿园', '小学', '初中', '高中']
    # Filter existing stages
    stages_exist = [s for s in stages if s in stage_tool.index]
    stage_tool = stage_tool.loc[stages_exist, top_n_tools] # reorder
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(stage_tool, annot=True, fmt='d', cmap='Oranges', linewidths=.5)
    plt.title('图6 学段与头部AI产品联动应用热力图', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('AI产品', fontsize=12)
    plt.ylabel('学段', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_a07_stage_product.png')
    plt.close()

    # 3. Subject x Product Heatmap
    subj_tool = pd.crosstab(df_plot['学科'].fillna('未知'), df_plot['tools_list'])
    subjs = ['语文', '数学', '英语', '科学', '美术']
    subjs_exist = [s for s in subjs if s in subj_tool.index]
    subj_tool = subj_tool.loc[subjs_exist, top_n_tools] # reorder
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(subj_tool, annot=True, fmt='d', cmap='Greens', linewidths=.5)
    plt.title('图7 核心学科与头部AI产品联动应用热力图', pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('AI产品', fontsize=12)
    plt.ylabel('学科', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_a08_subject_product.png')
    plt.close()

    print("Generated fig_a06, fig_a07, fig_a08 successfully.")

if __name__ == "__main__":
    process()
