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


# PLACEHOLDER_PART3
