#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "new_reviews" / "V5.xlsx"

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
    
    # Tools are already flat per row in V5
    df['tools_list'] = df['工具标准名'].fillna('未知')
    df_tools = df[df['tools_list'] != '未知'].copy()
    
    with open("crosstabs_result.txt", "w", encoding="utf-8") as f:
        f.write("\n--- 区域 X 产品联动 ---\n")
        region_tool = pd.crosstab(df_tools['区域'], df_tools['tools_list'])
        for r in ['东部', '中部', '西部']:
            if r in region_tool.index:
                top_tools = region_tool.loc[r].sort_values(ascending=False).head(5)
                f.write(f"[{r}地区] Top 5 工具: {', '.join([f'{k}({v})' for k,v in top_tools.items()])}\n")
                
        f.write("\n--- 学段 X 产品联动 ---\n")
        main_stages = ['幼儿园', '小学', '初中', '高中']
        stage_tool = pd.crosstab(df_tools['学段_标准化'], df_tools['tools_list'])
        for s in main_stages:
            if s in stage_tool.index:
                top_tools = stage_tool.loc[s].sort_values(ascending=False).head(5)
                f.write(f"[{s}] Top 5 工具: {', '.join([f'{k}({v})' for k,v in top_tools.items()])}\n")
                
        f.write("\n--- 学科 X 产品联动 ---\n")
        subj_tool = pd.crosstab(df_tools['学科'].fillna('未知'), df_tools['tools_list'])
        subjs = df_tools['学科'].value_counts()
        valid_subjs = [s for s in subjs.index if str(s) not in ['未提及', '未知', 'nan']][:5]
        for s in valid_subjs:
            if s in subj_tool.index:
                top_tools = subj_tool.loc[s].sort_values(ascending=False).head(5)
                f.write(f"[{s}] Top 5 工具: {', '.join([f'{k}({v})' for k,v in top_tools.items()])}\n")

if __name__ == "__main__":
    process()
