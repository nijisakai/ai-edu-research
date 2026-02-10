#!/usr/bin/env python3
"""
relabel_framework.py
Applies theoretical framework dimensions to all 3815 records in the education CSV.

New columns added:
  1. 三赋能分类 (Three Empowerments)
  2. iSTAR人机协同层级 (iSTAR Collaboration Level)
  3. D1_深度学习, D2_绿色鲁棒, D3_循证教学, D4_人机互信 (Digital Pedagogy binary)
  4. 智慧教育境界 (Smart Education Realm)
  5. 技术路径类型 (Technology Pathway Type)
  6. 创新深度评分 (Innovation Depth Score, 1-5)
  7. 产品技术代际 (Product Technology Generation)
"""

import pandas as pd
import numpy as np
import json
import re
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_IN = "/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv"
OUT_DIR = "/Users/sakai/Desktop/产业调研/ai-edu-research/output"
CSV_OUT = os.path.join(OUT_DIR, "教育产品统计_V6_框架标注.csv")
STATS_OUT = os.path.join(OUT_DIR, "framework_stats.json")
CASE_JSON = os.path.join(OUT_DIR, "case_deep_analysis.json")

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------
LLM_KEYWORDS = [
    "豆包", "DeepSeek", "Kimi", "文心一言", "通义千问", "ChatGPT", "Chat GPT",
    "GPT-4", "GPT4", "星火大模型", "讯飞星火", "智谱清言", "腾讯元宝", "元宝",
    "腾讯混元", "百川", "Qwen", "Claude", "Gemini", "千帆", "智谱",
    "央馆领航AI", "之江汇AI", "海螺AI", "扣子AI", "豆包智能体",
    "Kimi AI", "Kimi大模型", "Kimi智能助手",
]

MULTIMODAL_KEYWORDS = [
    "即梦", "Suno", "SunoAI", "Midjourney", "Stable Diffusion", "DALL",
    "可灵AI", "AI视频生成", "AI作图", "AI绘画", "AI音乐", "数字人",
    "剪映AI", "AI配音", "AI动画",
]

VR_AR_KEYWORDS = [
    "VR", "AR", "XR", "MR", "元宇宙", "虚拟实验", "虚拟仿真",
    "3D建模", "3D打印", "全景", "数字孪生",
]

HARDWARE_KEYWORDS = [
    "机器人", "传感器", "手环", "手表", "智能硬件", "智慧设备",
    "DIS", "mBot", "toio", "无人机", "3D打印机",
]

PLATFORM_KEYWORDS = [
    "教育平台", "管理系统", "智慧校园", "国家智慧教育平台", "希沃",
    "钉钉", "飞书", "问卷星", "智学网", "班级优化大师", "易课堂",
    "雨课堂", "学习通",
]

ASSESSMENT_KEYWORDS = [
    "评测", "批改", "诊断", "测评", "评估", "阅卷", "考试",
    "口语测评", "听说", "评分",
]

ANALYTICS_KEYWORDS = [
    "学情分析", "大数据", "精准教学", "数据分析", "画像", "预警",
    "数据挖掘", "学习分析",
]

# Traditional ICT tools
GEN1_KEYWORDS = [
    "PPT", "录播", "电子白板", "投影", "Word", "Excel", "PowerPoint",
]

# Internet+ tools
GEN2_KEYWORDS = [
    "在线教育", "MOOC", "直播", "微课", "慕课", "网课",
    "问卷星", "腾讯会议", "钉钉", "飞书",
]

# AI-assisted (non-LLM)
GEN3_KEYWORDS = [
    "智能批改", "自适应", "智能推荐", "OCR", "语音识别",
    "图像识别", "人脸识别", "智能评测",
]

# Deep learning evidence keywords
DEEP_LEARNING_KW = [
    "探究", "项目式", "创造", "批判性思维", "高阶思维", "深度学习",
    "问题解决", "协作", "创新", "设计思维", "跨学科", "STEAM",
    "PBL", "UbD", "逆向设计",
]

# Green/robust environment keywords
GREEN_ROBUST_KW = [
    "平台建设", "环境搭建", "基础设施", "网络安全", "数据安全",
    "隐私保护", "系统部署", "技术架构", "云平台", "数据治理",
]

# Evidence-based teaching keywords
EVIDENCE_KW = [
    "数据分析", "学情诊断", "精准教学", "评价反馈", "循证",
    "数据驱动", "学习分析", "过程性评价", "形成性评价", "画像",
]

# Human-machine trust keywords
TRUST_KW = [
    "人机协同", "AI助教", "智能导师", "人机互动", "人机交互",
    "AI伙伴", "AI助手", "协同教学", "人机共教",
]


# ---------------------------------------------------------------------------
# Helper: keyword match
# ---------------------------------------------------------------------------
def _contains_any(text, keywords):
    """Return True if text contains any of the keywords."""
    if pd.isna(text):
        return False
    text = str(text)
    return any(kw in text for kw in keywords)


def _match_tool(tool_name, keywords):
    """Check if a tool name matches any keyword."""
    if pd.isna(tool_name):
        return False
    t = str(tool_name)
    return any(kw in t for kw in keywords)


# ---------------------------------------------------------------------------
# 1. 三赋能分类
# ---------------------------------------------------------------------------
def assign_empowerment(row):
    l1 = str(row.get("应用场景（一级）", ""))
    l2 = str(row.get("应用场景（二级）", ""))
    main_scene = str(row.get("主要应用场景", ""))

    if l1 == "助管":
        return "赋能学校"
    if l1 == "助评":
        return "赋能评价"
    if l1 == "助教" or l1 == "助研":
        return "赋能教师"
    if l1 == "助学" or l1 == "助育":
        return "赋能学生"

    # Fallback from L2 / main scene
    mgmt_kw = ["校园", "安全", "行政", "管理", "监控", "信息管理"]
    eval_kw = ["评价", "评估", "评测", "考试"]
    teach_kw = ["备课", "教研", "教学设计", "教师"]
    if any(k in l2 for k in mgmt_kw) or any(k in main_scene for k in mgmt_kw):
        return "赋能学校"
    if any(k in l2 for k in eval_kw) or any(k in main_scene for k in eval_kw):
        return "赋能评价"
    if any(k in l2 for k in teach_kw) or any(k in main_scene for k in teach_kw):
        return "赋能教师"
    return "赋能学生"


# ---------------------------------------------------------------------------
# 2. iSTAR人机协同层级
# ---------------------------------------------------------------------------
def assign_istar(row):
    l1 = str(row.get("应用场景（一级）", ""))
    l2 = str(row.get("应用场景（二级）", ""))
    tool = str(row.get("工具标准名", ""))
    product_name = str(row.get("工具名称", ""))
    main_scene = str(row.get("主要应用场景", ""))

    # HUM(0): management/monitoring, no teaching interaction
    if l1 == "助管":
        return "HUM(0)"
    mgmt_l2 = ["校园安全", "学生信息", "教务管理", "智慧校园管理"]
    if any(k in l2 for k in mgmt_l2):
        return "HUM(0)"

    # Check if tool is an interactive LLM
    is_llm = _match_tool(tool, LLM_KEYWORDS) or _match_tool(product_name, LLM_KEYWORDS)

    # HM2C(2): student directly interacts with AI
    if l1 in ("助学", "助育"):
        if is_llm:
            return "HM2C(2)"
        # Interactive scenarios
        interactive_l2 = ["智能辅导系统", "情境式学习", "游戏化学习",
                          "智能阅读辅助", "智能心理支持"]
        if any(k in l2 for k in interactive_l2):
            # Check if tool is chatbot-like
            chatbot_kw = ["对话", "聊天", "助手", "智能体", "AI互动", "语音互动"]
            if any(k in tool for k in chatbot_kw) or any(k in product_name for k in chatbot_kw):
                return "HM2C(2)"
            # Multimodal AI tools also count as student-interactive
            if _match_tool(tool, MULTIMODAL_KEYWORDS):
                return "HM2C(2)"
            return "HMC(1)"
        # Non-interactive student tools
        return "HMC(1)"

    # HMC(1): teacher uses AI, student doesn't directly interact
    if l1 in ("助教", "助研"):
        return "HMC(1)"

    if l1 == "助评":
        return "HMC(1)"

    # Default
    return "HMC(1)"



# ---------------------------------------------------------------------------
# 3. 数字教学法维度 (binary columns)
# ---------------------------------------------------------------------------
def assign_pedagogy_dims(row):
    """Return dict with D1-D4 binary flags."""
    innovation = str(row.get("优势和创新点", ""))
    innovation2 = str(row.get("优势和创新点.1", ""))
    tech = str(row.get("关键技术路径", ""))
    l2 = str(row.get("应用场景（二级）", ""))
    effect = str(row.get("潜在成效", ""))
    combined = f"{innovation} {innovation2} {tech} {l2} {effect}"

    d1 = int(_contains_any(combined, DEEP_LEARNING_KW))
    d2 = int(_contains_any(combined, GREEN_ROBUST_KW))
    d3 = int(_contains_any(combined, EVIDENCE_KW))
    d4 = int(_contains_any(combined, TRUST_KW))

    # If none matched, try to infer from scenario
    if d1 + d2 + d3 + d4 == 0:
        if "辅导" in l2 or "学习" in l2:
            d1 = 1
        elif "分析" in l2 or "诊断" in l2:
            d3 = 1

    return {"D1_深度学习": d1, "D2_绿色鲁棒": d2, "D3_循证教学": d3, "D4_人机互信": d4}


# ---------------------------------------------------------------------------
# 4. 智慧教育境界
# ---------------------------------------------------------------------------
def assign_smart_realm(row):
    product_type = str(row.get("产品分类", ""))
    product_form = str(row.get("产品形态", ""))
    innovation = str(row.get("优势和创新点", ""))
    innovation2 = str(row.get("优势和创新点.1", ""))
    l1 = str(row.get("应用场景（一级）", ""))
    l2 = str(row.get("应用场景（二级）", ""))
    combined = f"{innovation} {innovation2} {l2}"

    # 第三境界: institutional change
    realm3_kw = ["评价改革", "课程重构", "管理创新", "制度", "体制",
                 "评价体系", "课程体系", "教育治理", "区域级"]
    if any(k in combined for k in realm3_kw) or "校级 / 区域级" in product_type:
        return "第三境界_制度变革"

    # 第二境界: new teaching models
    realm2_kw = ["翻转课堂", "混合式", "项目式", "PBL", "探究式",
                 "教学模式", "教学范式", "新样态", "新模式", "STEAM",
                 "跨学科", "深度学习", "协作学习"]
    if any(k in combined for k in realm2_kw):
        return "第二境界_教学模式"

    # 第一境界: environment/infrastructure
    realm1_kw = ["硬件", "平台", "系统", "环境", "基础设施", "设备",
                 "部署", "搭建"]
    if l1 == "助管":
        return "第一境界_智慧环境"
    if any(k in product_form for k in ["硬件", "AI系统"]):
        return "第一境界_智慧环境"
    if any(k in combined for k in realm1_kw):
        return "第一境界_智慧环境"

    # Default: most AI-in-education cases are about teaching models
    return "第二境界_教学模式"



# ---------------------------------------------------------------------------
# 5. 技术路径类型
# ---------------------------------------------------------------------------
def assign_tech_pathway(row):
    tool = str(row.get("工具标准名", ""))
    product_name = str(row.get("工具名称", ""))
    product_type = str(row.get("产品分类", ""))
    product_form = str(row.get("产品形态", ""))
    combined_tool = f"{tool} {product_name}"

    # T4: immersive (VR/AR)
    if _match_tool(combined_tool, VR_AR_KEYWORDS):
        return "T4_沉浸体验"

    # T5: hardware
    if _match_tool(combined_tool, HARDWARE_KEYWORDS):
        return "T5_智能硬件"
    if "硬件" in product_form:
        return "T5_智能硬件"

    # T1: content generation (LLM-based)
    if _match_tool(combined_tool, LLM_KEYWORDS):
        return "T1_内容生成"

    # T1 also: multimodal generation
    if _match_tool(combined_tool, MULTIMODAL_KEYWORDS):
        return "T1_内容生成"

    # T2: assessment
    if _match_tool(combined_tool, ASSESSMENT_KEYWORDS):
        return "T2_智能评测"
    l2 = str(row.get("应用场景（二级）", ""))
    if any(k in l2 for k in ["评价", "评测", "批改", "诊断", "评估"]):
        return "T2_智能评测"

    # T3: data-driven
    if _match_tool(combined_tool, ANALYTICS_KEYWORDS):
        return "T3_数据驱动"
    if any(k in l2 for k in ["学情分析", "数据分析", "学情诊断"]):
        return "T3_数据驱动"

    # T6: platform ecosystem
    if _match_tool(combined_tool, PLATFORM_KEYWORDS):
        return "T6_平台生态"
    if "平台" in product_type or "SaaS" in product_type:
        return "T6_平台生态"

    # Default based on product type
    if "AI" in product_type:
        return "T1_内容生成"
    return "T6_平台生态"


# ---------------------------------------------------------------------------
# 6. 创新深度评分 (1-5)
# ---------------------------------------------------------------------------
def compute_innovation_score(row, pedagogy_dims):
    score = 1  # base score

    # Number of unique tools per case - handled at case level later
    # For now, check self-developed
    if row.get("是否自主研发") == True or str(row.get("是否自主研发")) == "True":
        score += 1

    # iSTAR level bonus
    istar = str(row.get("iSTAR人机协同层级", ""))
    if "HM2C" in istar:
        score += 1
    elif "HMnC" in istar:
        score += 2

    # Pedagogy dimensions covered
    dims_count = sum([pedagogy_dims.get("D1_深度学习", 0),
                      pedagogy_dims.get("D2_绿色鲁棒", 0),
                      pedagogy_dims.get("D3_循证教学", 0),
                      pedagogy_dims.get("D4_人机互信", 0)])
    if dims_count >= 2:
        score += 1

    return min(score, 5)



# ---------------------------------------------------------------------------
# 7. 产品技术代际
# ---------------------------------------------------------------------------
def assign_tech_generation(row):
    tool = str(row.get("工具标准名", ""))
    product_name = str(row.get("工具名称", ""))
    combined = f"{tool} {product_name}"

    # Gen5: multimodal AI
    if _match_tool(combined, MULTIMODAL_KEYWORDS):
        return "Gen5_多模态AI"

    # Gen4: LLM-based
    if _match_tool(combined, LLM_KEYWORDS):
        return "Gen4_大模型"

    # Gen1: traditional ICT
    if _match_tool(combined, GEN1_KEYWORDS):
        return "Gen1_传统信息化"

    # Gen2: internet+
    if _match_tool(combined, GEN2_KEYWORDS):
        return "Gen2_互联网+"

    # Gen3: AI-assisted (non-LLM)
    if _match_tool(combined, GEN3_KEYWORDS):
        return "Gen3_AI辅助"

    # Infer from product type
    product_type = str(row.get("产品分类", ""))
    if "AI" in product_type:
        return "Gen3_AI辅助"
    if "SaaS" in product_type or "Web" in product_type:
        return "Gen2_互联网+"
    if "平台" in product_type:
        return "Gen2_互联网+"

    # Default: most tools in this dataset are AI-assisted
    return "Gen3_AI辅助"


# ---------------------------------------------------------------------------
# Main: load, label, save
# ---------------------------------------------------------------------------
def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_IN, encoding="utf-8-sig")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Load case deep analysis for iSTAR overrides
    istar_overrides = {}
    if os.path.exists(CASE_JSON):
        with open(CASE_JSON, "r", encoding="utf-8-sig") as f:
            case_data = json.load(f)
        for c in case_data.get("cases", []):
            cid = c.get("case_id", "")
            # Extract numeric ID
            m = re.search(r"(\d+)", cid)
            if m:
                case_num = int(m.group(1))
                istar_overrides[case_num] = c.get("istar_level", "")
        print(f"  Loaded {len(istar_overrides)} iSTAR overrides from case analysis")

    # Count tools per case for innovation score
    tools_per_case = df.groupby("案例编号")["工具标准名"].nunique().to_dict()

    # -----------------------------------------------------------------------
    # Apply all labels
    # -----------------------------------------------------------------------
    print("Applying framework labels...")

    # 1. 三赋能分类
    df["三赋能分类"] = df.apply(assign_empowerment, axis=1)

    # 2. iSTAR
    df["iSTAR人机协同层级"] = df.apply(assign_istar, axis=1)
    # Override with deep analysis where available
    for idx, row in df.iterrows():
        case_num = row.get("案例编号")
        if case_num in istar_overrides and istar_overrides[case_num]:
            override = istar_overrides[case_num]
            # Map to our format
            level_map = {
                "HUM": "HUM(0)", "HMC": "HMC(1)",
                "HM2C": "HM2C(2)", "HMnC": "HMnC(3)",
            }
            mapped = level_map.get(override, None)
            if mapped:
                df.at[idx, "iSTAR人机协同层级"] = mapped

    # 3. Digital pedagogy dimensions
    pedagogy_results = df.apply(assign_pedagogy_dims, axis=1, result_type="expand")
    df["D1_深度学习"] = pedagogy_results["D1_深度学习"]
    df["D2_绿色鲁棒"] = pedagogy_results["D2_绿色鲁棒"]
    df["D3_循证教学"] = pedagogy_results["D3_循证教学"]
    df["D4_人机互信"] = pedagogy_results["D4_人机互信"]

    # 4. 智慧教育境界
    df["智慧教育境界"] = df.apply(assign_smart_realm, axis=1)

    # 5. 技术路径类型
    df["技术路径类型"] = df.apply(assign_tech_pathway, axis=1)

    # 6. 创新深度评分
    innovation_scores = []
    for idx, row in df.iterrows():
        pdims = {
            "D1_深度学习": df.at[idx, "D1_深度学习"],
            "D2_绿色鲁棒": df.at[idx, "D2_绿色鲁棒"],
            "D3_循证教学": df.at[idx, "D3_循证教学"],
            "D4_人机互信": df.at[idx, "D4_人机互信"],
        }
        base = compute_innovation_score(row, pdims)
        # Add tool count bonus
        case_num = row.get("案例编号")
        n_tools = tools_per_case.get(case_num, 1)
        if n_tools >= 2:
            base = min(base + 1, 5)
        innovation_scores.append(base)
    df["创新深度评分"] = innovation_scores

    # 7. 产品技术代际
    df["产品技术代际"] = df.apply(assign_tech_generation, axis=1)

    # -----------------------------------------------------------------------
    # Save enhanced CSV
    # -----------------------------------------------------------------------
    print(f"\nSaving enhanced CSV to {CSV_OUT}...")
    df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} rows, {len(df.columns)} columns")

    # -----------------------------------------------------------------------
    # Compute and save framework statistics
    # -----------------------------------------------------------------------
    print("\nComputing framework statistics...")
    stats = {}

    # 三赋能分类
    stats["三赋能分类"] = df["三赋能分类"].value_counts().to_dict()

    # iSTAR
    stats["iSTAR人机协同层级"] = df["iSTAR人机协同层级"].value_counts().to_dict()

    # Digital pedagogy
    stats["数字教学法维度"] = {
        "D1_深度学习": int(df["D1_深度学习"].sum()),
        "D2_绿色鲁棒": int(df["D2_绿色鲁棒"].sum()),
        "D3_循证教学": int(df["D3_循证教学"].sum()),
        "D4_人机互信": int(df["D4_人机互信"].sum()),
    }

    # 智慧教育境界
    stats["智慧教育境界"] = df["智慧教育境界"].value_counts().to_dict()

    # 技术路径类型
    stats["技术路径类型"] = df["技术路径类型"].value_counts().to_dict()

    # 创新深度评分
    stats["创新深度评分"] = df["创新深度评分"].value_counts().sort_index().to_dict()
    stats["创新深度评分_mean"] = round(float(df["创新深度评分"].mean()), 2)

    # 产品技术代际
    stats["产品技术代际"] = df["产品技术代际"].value_counts().to_dict()

    # Cross-tabulations
    stats["三赋能_x_iSTAR"] = pd.crosstab(
        df["三赋能分类"], df["iSTAR人机协同层级"]
    ).to_dict()
    stats["三赋能_x_技术代际"] = pd.crosstab(
        df["三赋能分类"], df["产品技术代际"]
    ).to_dict()
    stats["iSTAR_x_技术路径"] = pd.crosstab(
        df["iSTAR人机协同层级"], df["技术路径类型"]
    ).to_dict()

    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Saved statistics to {STATS_OUT}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FRAMEWORK LABELING SUMMARY")
    print("=" * 60)

    new_cols = ["三赋能分类", "iSTAR人机协同层级", "D1_深度学习", "D2_绿色鲁棒",
                "D3_循证教学", "D4_人机互信", "智慧教育境界", "技术路径类型",
                "创新深度评分", "产品技术代际"]
    print(f"\nNew columns added: {len(new_cols)}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    for col in new_cols:
        print(f"\n--- {col} ---")
        if col in ["D1_深度学习", "D2_绿色鲁棒", "D3_循证教学", "D4_人机互信"]:
            print(f"  1 (yes): {int(df[col].sum())}  |  0 (no): {int((df[col] == 0).sum())}")
        elif col == "创新深度评分":
            print(df[col].value_counts().sort_index().to_string())
            print(f"  Mean: {df[col].mean():.2f}")
        else:
            print(df[col].value_counts().to_string())

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
