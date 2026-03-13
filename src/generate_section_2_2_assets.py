from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from matplotlib import font_manager
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PptInches
from pptx.util import Pt as PptPt


BASE_DIR = Path(__file__).resolve().parent.parent
WORKBOOK_PATH = BASE_DIR / "data" / "V6.xlsx"
DOCX_PATH = BASE_DIR / "Section_2.2_应用现状综述_最终版.docx"
PPTX_PATH = BASE_DIR / "六大场景统计_区域学科学段_两页PPT.pptx"
SUMMARY_PATH = BASE_DIR / "output" / "V6_workbook_sheet_summary.md"
FIG_DIR = BASE_DIR / "output" / "figures" / "section_2_2_v6"

SCENE_ORDER = ["助学", "助教", "助评", "助育", "助管", "助研"]
SCENE_COLORS = {
    "助学": "#4C78A8",
    "助教": "#F58518",
    "助评": "#E45756",
    "助育": "#72B7B2",
    "助管": "#54A24B",
    "助研": "#B279A2",
}


@dataclass
class AnalysisBundle:
    sheets: list[dict]
    df: pd.DataFrame
    cases: pd.DataFrame
    region_scene: pd.DataFrame
    stage_scene: pd.DataFrame
    subject_scene: pd.DataFrame
    summary: dict


def configure_matplotlib() -> None:
    preferred_font_names = [
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "Songti SC",
        "Heiti TC",
        "PingFang HK",
        "DejaVu Sans",
    ]
    available_names = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((name for name in preferred_font_names if name in available_names), "DejaVu Sans")
    plt.rcParams["font.family"] = [selected_font]
    plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", rc={"font.family": selected_font, "axes.unicode_minus": False})


def normalize_text(value) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text in {"", "nan", "None", "未提及", "未知"}:
        return pd.NA
    return text


def normalize_product_name(value) -> str | pd.NA:
    text = normalize_text(value)
    if pd.isna(text):
        return pd.NA
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9])\s+(?=AI\b)", "", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    alias_map = {
        "即梦": "即梦AI",
        "即梦 AI": "即梦AI",
        "剪映": "剪映AI",
        "剪映 AI": "剪映AI",
        "DeepSeek": "DeepSeek 大模型",
        "语文朗读宝 AI": "语文朗读宝AI",
    }
    return alias_map.get(text, text)


def normalize_stage(value) -> str:
    if pd.isna(value):
        return "未提及"
    text = str(value)
    if "小学/初中/高中" in text:
        return "全学段"
    if "小学至初中" in text:
        return "小学/初中"
    if "初中/高中" in text:
        return "初中/高中"
    if "幼儿园" in text or "学前" in text:
        return "学前"
    if "小学" in text:
        return "小学"
    if "初中" in text:
        return "初中"
    if "高中" in text:
        return "高中"
    if "中学" in text:
        return "中学"
    if "中职" in text or "职高" in text:
        return "中职"
    return text.strip() or "未提及"


def pct(part: float, whole: float) -> str:
    if not whole:
        return "0.0%"
    return f"{part / whole * 100:.1f}%"


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.font.size = Pt(10)
    run.bold = True


def insert_picture(doc: Document, path: Path, width: float = 5.8) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))


def workbook_role(sheet_name: str) -> str:
    mapping = {
        "（新）processed_results": "主事实表：3815条工具级记录，承载案例、场景、学科、省份与产品分类字段",
        "省份产品": "省份-产品展开表，用于产品地理分布追踪",
        "Sheet1": "案例级摘要表，用于理解1690+案例的压缩口径",
        "Sheet2": "产品频次明细与透视辅助表",
        "「工具标准名」映射表": "原始工具名称到标准产品名/场景/学科的映射字典",
        "产品分类_教育_参考表": "教育技术类型参考表",
        "产品分类_产业_参考表": "AI产业链分类参考表",
        "省份映 射": "地区到省份的补充映射表",
        "省份映射": "地区到省份的补充映射表",
        "Sheet4": "省级城市/区县覆盖汇总表",
        "Top 40产品": "高频产品的产业分类、场景分类与技术分类总表",
        "Top40产品-场景简介": "头部产品的场景简介与应用说明",
        "Top": "头部产品清单",
        "参考-三类产品分类总表": "产业分类总参考",
        "2. 场景驱动分类-表格": "六大一级场景与29个二级场景字典",
        "1. 产业链分类-表格": "产业链分类表格版",
        "3. 教育领域分类-表格": "教育领域技术分类表格版",
        "助育、助管、助评、助研（雅溶&杨妍）": "长尾场景补充说明表",
    }
    return mapping.get(sheet_name, "辅助工作表")


def read_workbook_summary(path: Path) -> list[dict]:
    excel = pd.ExcelFile(path)
    sheets = []
    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        sheets.append(
            {
                "name": sheet_name,
                "shape": frame.shape,
                "columns": list(frame.columns),
                "role": workbook_role(sheet_name),
            }
        )
    return sheets


def load_bundle() -> AnalysisBundle:
    sheets = read_workbook_summary(WORKBOOK_PATH)
    df = pd.read_excel(WORKBOOK_PATH, sheet_name="（新）processed_results")
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(normalize_text)

    df["案例编号"] = pd.to_numeric(df["案例编号"], errors="coerce")
    df["省份_标准"] = df["省份_更新"].fillna(df["省份"])
    df["产品名_标准"] = df["产品名（校准）"].fillna(df["工具标准名"]).map(normalize_product_name)

    cases = df.dropna(subset=["案例编号"]).sort_values("案例编号").drop_duplicates("案例编号", keep="first").copy()
    cases["学段_标准"] = cases["学段"].apply(normalize_stage)

    province_region = {
        "北京市": "东部",
        "天津市": "东部",
        "河北省": "东部",
        "上海市": "东部",
        "江苏省": "东部",
        "浙江省": "东部",
        "福建省": "东部",
        "山东省": "东部",
        "广东省": "东部",
        "海南省": "东部",
        "辽宁省": "东部",
        "山西省": "中部",
        "吉林省": "中部",
        "黑龙江省": "中部",
        "安徽省": "中部",
        "江西省": "中部",
        "河南省": "中部",
        "湖北省": "中部",
        "湖南省": "中部",
        "内蒙古自治区": "西部",
        "广西壮族自治区": "西部",
        "重庆市": "西部",
        "四川省": "西部",
        "贵州省": "西部",
        "云南省": "西部",
        "西藏自治区": "西部",
        "陕西省": "西部",
        "甘肃省": "西部",
        "青海省": "西部",
        "宁夏回族自治区": "西部",
        "新疆维吾尔自治区": "西部",
    }
    cases["区域"] = cases["省份_标准"].map(province_region).fillna("未提及")

    region_scene = pd.crosstab(cases["区域"], cases["应用场景（一级）"])
    region_scene = region_scene.reindex(index=["东部", "中部", "西部", "未提及"], fill_value=0)
    region_scene = region_scene.reindex(columns=SCENE_ORDER, fill_value=0)

    stage_order = ["学前", "小学", "初中", "高中", "中学", "小学/初中", "初中/高中", "全学段", "中职", "未提及"]
    stage_scene = pd.crosstab(cases["学段_标准"], cases["应用场景（一级）"])
    stage_scene = stage_scene.reindex(index=[item for item in stage_order if item in stage_scene.index], fill_value=0)
    stage_scene = stage_scene.reindex(columns=SCENE_ORDER, fill_value=0)

    subject_counts = cases["学科"].fillna("未提及").value_counts()
    subject_order = [subject for subject in subject_counts.index if subject != "未提及"][:10]
    subject_scene = pd.crosstab(cases[cases["学科"].isin(subject_order)]["学科"], cases[cases["学科"].isin(subject_order)]["应用场景（一级）"])
    subject_scene = subject_scene.reindex(index=subject_order, fill_value=0)
    subject_scene = subject_scene.reindex(columns=SCENE_ORDER, fill_value=0)

    l1_counts = cases["应用场景（一级）"].value_counts().reindex(SCENE_ORDER, fill_value=0)
    l2_pairs = (
        cases.dropna(subset=["应用场景（一级）", "应用场景（二级）"])
        .groupby(["应用场景（一级）", "应用场景（二级）"])
        .size()
        .reset_index(name="count")
        .sort_values(["应用场景（一级）", "count"], ascending=[True, False])
    )
    top_l2_by_l1 = {}
    for scene in SCENE_ORDER:
        subset = l2_pairs[l2_pairs["应用场景（一级）"] == scene].head(2)
        top_l2_by_l1[scene] = [f"{row['应用场景（二级）']}（{int(row['count'])}例）" for _, row in subset.iterrows()]

    summary = {
        "sheet_count": len(sheets),
        "tool_rows": int(len(df)),
        "case_count": int(cases["案例编号"].nunique()),
        "product_count": int(df["产品名_标准"].dropna().nunique()),
        "company_count": int(df["公司"].dropna().nunique()),
        "province_count": int(cases["省份_标准"].dropna().nunique()),
        "stage_counts": cases["学段_标准"].value_counts().to_dict(),
        "subject_counts": cases["学科"].fillna("未提及").value_counts().to_dict(),
        "province_counts": cases["省份_标准"].fillna("未提及").value_counts().to_dict(),
        "region_counts": cases["区域"].value_counts().to_dict(),
        "scene_counts": l1_counts.to_dict(),
        "top_l2_by_l1": top_l2_by_l1,
        "top_products": df["产品名_标准"].dropna().value_counts().head(15).to_dict(),
    }

    return AnalysisBundle(
        sheets=sheets,
        df=df,
        cases=cases,
        region_scene=region_scene,
        stage_scene=stage_scene,
        subject_scene=subject_scene,
        summary=summary,
    )


def save_workbook_summary(bundle: AnalysisBundle) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V6.xlsx 工作簿结构摘要",
        "",
        f"- 工作表总数：{bundle.summary['sheet_count']}",
        f"- 主事实表：`（新）processed_results`，共 {bundle.summary['tool_rows']} 条工具级记录",
        f"- 案例口径：{bundle.summary['case_count']} 个去重案例",
        "",
        "| Sheet | 规模 | 作用 |",
        "|---|---:|---|",
    ]
    for sheet in bundle.sheets:
        lines.append(f"| {sheet['name']} | {sheet['shape'][0]}×{sheet['shape'][1]} | {sheet['role']} |")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def barh_chart(series: pd.Series, title: str, output: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ordered = series.sort_values(ascending=True)
    ax.barh(ordered.index, ordered.values, color=color)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("案例数")
    for idx, value in enumerate(ordered.values):
        ax.text(value + max(ordered.values) * 0.01, idx, str(int(value)), va="center", fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def scene_bar_chart(series: pd.Series, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [SCENE_COLORS[item] for item in series.index]
    bars = ax.bar(series.index, series.values, color=colors)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_ylabel("案例数")
    for bar, value in zip(bars, series.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(int(value)), ha="center", va="bottom", fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def stacked_region_chart(df: pd.DataFrame, title: str, output: Path) -> None:
    plot_df = df.loc[[idx for idx in df.index if idx != "未提及"]].copy()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    left = pd.Series(0, index=plot_df.index, dtype=float)
    for scene in SCENE_ORDER:
        values = plot_df[scene]
        ax.barh(plot_df.index, values, left=left, color=SCENE_COLORS[scene], label=scene)
        left += values
    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel("案例数")
    ax.legend(ncol=6, bbox_to_anchor=(0.5, 1.12), loc="upper center", frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def heatmap_chart(df: pd.DataFrame, title: str, output: Path) -> None:
    width = max(8.5, 1.2 * len(df.columns) + 3.5)
    height = max(4.8, 0.55 * len(df.index) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(df, annot=True, fmt="g", cmap="YlGnBu", linewidths=0.5, cbar=True, ax=ax)
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def generate_figures(bundle: AnalysisBundle) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "province": FIG_DIR / "fig_section22_province_top10.png",
        "stage": FIG_DIR / "fig_section22_stage_distribution.png",
        "subject": FIG_DIR / "fig_section22_subject_top10.png",
        "scene": FIG_DIR / "fig_section22_scene_l1.png",
        "region_scene": FIG_DIR / "fig_section22_region_scene.png",
        "stage_scene": FIG_DIR / "fig_section22_stage_scene_heatmap.png",
        "subject_scene": FIG_DIR / "fig_section22_subject_scene_heatmap.png",
    }
    province_series = pd.Series(bundle.summary["province_counts"]).head(10)
    stage_series = pd.Series(bundle.summary["stage_counts"]).head(8)
    subject_series = pd.Series(bundle.summary["subject_counts"]).drop(labels=["未提及"], errors="ignore").head(10)
    scene_series = pd.Series(bundle.summary["scene_counts"]).reindex(SCENE_ORDER).fillna(0)

    barh_chart(province_series, "V6 案例量最高的 10 个省份", figure_paths["province"], "#4C78A8")
    barh_chart(stage_series, "V6 案例学段分布", figure_paths["stage"], "#72B7B2")
    barh_chart(subject_series, "V6 案例学科 Top10", figure_paths["subject"], "#F58518")
    scene_bar_chart(scene_series, "V6 六大一级应用场景分布", figure_paths["scene"])
    stacked_region_chart(bundle.region_scene, "六大场景的区域分布", figure_paths["region_scene"])
    heatmap_chart(bundle.stage_scene, "六大场景 × 学段分布", figure_paths["stage_scene"])
    heatmap_chart(bundle.subject_scene, "六大场景 × 学科分布（Top10 学科）", figure_paths["subject_scene"])
    return figure_paths


def set_docx_style(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    style.font.size = Pt(11)


def add_paragraph(doc: Document, text: str, bold_prefix: str | None = None) -> None:
    paragraph = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        paragraph.add_run(text[len(bold_prefix):])
    else:
        paragraph.add_run(text)


def build_docx(bundle: AnalysisBundle, figures: dict[str, Path]) -> None:
    doc = Document()
    set_docx_style(doc)

    summary = bundle.summary
    total_cases = summary["case_count"]
    top_stage, top_stage_count = next(iter(summary["stage_counts"].items()))
    top_subject, top_subject_count = next(iter({k: v for k, v in summary["subject_counts"].items() if k != "未提及"}.items()))
    top_province, top_province_count = next(iter({k: v for k, v in summary["province_counts"].items() if k != "未提及"}.items()))
    scene_counts = pd.Series(summary["scene_counts"]).reindex(SCENE_ORDER).fillna(0)
    dominant_scene = scene_counts.idxmax()
    dominant_scene_count = int(scene_counts.max())

    east_count = int(bundle.region_scene.loc["东部"].sum())
    middle_count = int(bundle.region_scene.loc["中部"].sum())
    west_count = int(bundle.region_scene.loc["西部"].sum())

    doc.add_heading("2.2 教育应用类产品应用现状综述", level=1)
    add_paragraph(
        doc,
        f"基于 V6.xlsx 的全量更新，本节重新梳理了教育应用类产品的实践格局。新版工作簿共包含 {summary['sheet_count']} 个 sheets，其中主事实表“（新）processed_results”记录了 {summary['tool_rows']} 条工具级记录，并可回收为 {total_cases} 个案例、{summary['product_count']} 个标准化产品和 {summary['province_count']} 个省级地域单元。与旧版 V5 口径相比，V6 已经不只是底表替换，而是形成了“主事实表 + 场景分类字典 + 产品分类参考 + 头部产品说明 + 省份映射”的完整分析工作簿结构。"
    )

    doc.add_heading("2.2.1 数据基础与工作簿理解", level=2)
    add_paragraph(
        doc,
        f"从工作簿结构看，V6 将项目所需的五类信息整合在同一文件内：一是主事实表“（新）processed_results”，承载案例、产品、学科、场景、省份等核心变量；二是“省份产品”“Sheet1”“Sheet2”等展开表，用于在产品级与案例级之间切换口径；三是“2. 场景驱动分类-表格”“产品分类_教育_参考表”“产品分类_产业_参考表”等字典表，为场景、教育技术、产业链三套分类口径提供统一标准；四是“Top 40产品”“Top40产品-场景简介”“Top”等产品说明表，帮助识别高频产品及其典型应用；五是“省份映射”“Sheet4”等地域补充表，用于完善地区到省份的归一化。"
    )
    add_paragraph(
        doc,
        "这种结构说明，本项目并非单纯的案例汇编，而是围绕“案例事实、产品标准化、场景分类、产业分类、地域映射”建立了可复用的研究数据底座。Section 2.2 的更新因此不再沿用旧版写死数字，而是直接从 V6 的主事实表和字典表同步生成。"
    )

    doc.add_heading("2.2.2 学段学科角度：应用落地的主阵地", level=2)
    add_paragraph(
        doc,
        f"从案例级口径看，{top_stage}仍是 AI 教育应用最集中的落地学段，共 {top_stage_count} 例，占全部案例的 {pct(top_stage_count, total_cases)}。其后依次为初中和高中，说明当前实践仍主要围绕义务教育与升学关键阶段展开。跨学段或全学段案例虽已出现，但总体体量仍明显小于单一学段场景，反映出产品适配仍以具体教学组织单元为主，而非完全通用化部署。"
    )
    add_paragraph(
        doc,
        f"学科维度同样表现出显著集中性。{top_subject}以 {top_subject_count} 例位居首位，语文、英语、科学、物理等学科随后跟进，表明当前 AI 应用最容易切入的是既存在高频教学任务、又便于形成标准化反馈的数据密集型学科。与之相比，美术、体育、心理健康等素养型领域虽然总体体量较小，但在特定场景中已形成较强特色，说明项目正在从“主科增效”向“多学科延展”过渡。"
    )
    insert_picture(doc, figures["stage"])
    add_caption(doc, "图2-1 V6 案例学段分布")
    insert_picture(doc, figures["subject"])
    add_caption(doc, "图2-2 V6 案例学科 Top10")

    doc.add_heading("2.2.3 区域角度：东部领跑，中西部加速跟进", level=2)
    add_paragraph(
        doc,
        f"区域维度上，东部样本仍保持绝对领先，东部、中部、西部分别对应 {east_count}、{middle_count}、{west_count} 个案例。若按案例来源省份观察，{top_province}以 {top_province_count} 例位列第一，说明头部省份仍在以平台建设、课程整合和区域推进三种方式同步放大案例产出。"
    )
    add_paragraph(
        doc,
        "值得注意的是，中西部并未停留在零散试点状态。V6 中，西部样本已在助学、助教、助评、助育等多个场景形成可观分布，反映出 AI 应用正在从东部先行试验转向更广范围的跨区域扩散。区域差异仍然存在，但差异的核心已不再是“有没有”，而是“主要集中在哪些场景、以何种形态部署”。"
    )
    insert_picture(doc, figures["province"])
    add_caption(doc, "图2-3 V6 案例量最高的 10 个省份")

    doc.add_heading("2.2.4 场景角度：一级场景结构先行呈现", level=2)
    add_paragraph(
        doc,
        f"按照你的要求，本轮正文中的场景表先仅保留一级场景。V6 显示，{dominant_scene}仍然是最核心的应用方向，共 {dominant_scene_count} 例，占全部案例的 {pct(dominant_scene_count, total_cases)}。其后依次是助教、助评、助育、助管和助研，整体呈现“以学生学习支持为主轴、以教师赋能和评价改革为次主线、以治理与研究场景为长尾”的结构特征。"
    )
    add_paragraph(
        doc,
        "进一步看，一级场景之下的二级场景已经开始分化出相对清晰的功能重心：助学主要集中在智能辅导系统、情境式学习与智能作业设计；助教则聚焦教学分析、作业管理与课堂管理；助评明显向综合素质评价与学生评估集中；助育主要围绕智能心理支持和德育/家庭教育指导展开；助管与助研则保持较小规模，但已显现出学生信息管理、校园治理和科研辅助等明确切口。"
    )

    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    header = table.rows[0].cells
    header[0].text = "一级应用场景"
    header[1].text = "案例数"
    header[2].text = "占比"
    header[3].text = "核心二级场景提要"
    for scene in SCENE_ORDER:
        row = table.add_row().cells
        count = int(scene_counts[scene])
        row[0].text = scene
        row[1].text = str(count)
        row[2].text = pct(count, total_cases)
        row[3].text = "；".join(bundle.summary["top_l2_by_l1"].get(scene, [])) or "-"

    insert_picture(doc, figures["scene"])
    add_caption(doc, "图2-4 V6 六大一级应用场景分布")

    doc.add_heading("2.2.5 综合判断：从单点应用转向结构分化", level=2)
    add_paragraph(
        doc,
        "综合学段、学科、区域和场景四个维度，可以将当前 AI 教育应用概括为三点。第一，应用主阵地仍在小学与主干学科，但多学科扩散趋势已经出现。第二，东部仍是高密度集聚区，但中西部正在通过标准化平台和通用模型快速补位。第三，六大场景已不再是单纯的“助学独大”，而是开始沿着教师赋能、评价改革、心理支持和教育治理等方向展开结构性分化。"
    )
    add_paragraph(
        doc,
        "也正因为如此，本次额外生成的两页 PPT 不再重复总体分布，而是专门聚焦六大场景在区域、学段、学科上的交叉分布，用于把场景结构从“总体占比”进一步推进到“在哪里发生、由谁使用、集中在哪些教育单元”这一层。"
    )

    doc.save(DOCX_PATH)


def add_textbox(slide, left: float, top: float, width: float, height: float, text: str, font_size: int = 18, bold: bool = False) -> None:
    box = slide.shapes.add_textbox(PptInches(left), PptInches(top), PptInches(width), PptInches(height))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.size = PptPt(font_size)
    paragraph.font.bold = bold
    paragraph.font.name = "PingFang SC"


def build_ppt(bundle: AnalysisBundle, figures: dict[str, Path]) -> None:
    presentation = Presentation()
    presentation.slide_width = PptInches(13.333)
    presentation.slide_height = PptInches(7.5)

    scene_counts = pd.Series(bundle.summary["scene_counts"]).reindex(SCENE_ORDER).fillna(0)
    region_top = bundle.region_scene.loc[["东部", "中部", "西部"]]
    east_share = pct(region_top.loc["东部"].sum(), region_top.to_numpy().sum())
    dominant_stage = {scene: bundle.stage_scene[scene].idxmax() for scene in SCENE_ORDER if bundle.stage_scene[scene].sum() > 0}
    dominant_subject = {scene: bundle.subject_scene[scene].idxmax() for scene in SCENE_ORDER if bundle.subject_scene[scene].sum() > 0}

    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    add_textbox(slide, 0.45, 0.18, 12.3, 0.45, "六大场景统计：区域分布", 24, True)
    add_textbox(slide, 0.48, 0.62, 12.0, 0.35, f"数据口径：V6.xlsx 案例级去重样本 {bundle.summary['case_count']} 例；场景按一级分类统计", 11)
    slide.shapes.add_picture(str(figures["region_scene"]), PptInches(0.45), PptInches(1.05), width=PptInches(8.2))
    slide.shapes.add_picture(str(figures["scene"]), PptInches(8.95), PptInches(1.12), width=PptInches(3.95))
    insight_text = (
        f"1. 东部承载 {east_share} 的场景案例总量，是六大场景共同的主集聚区。\n"
        f"2. 场景总体仍以{scene_counts.idxmax()}为主，案例数 {int(scene_counts.max())}。\n"
        f"3. 中西部已在助教、助评、助育等场景形成可见扩散，不再只是单一助学渗透。"
    )
    add_textbox(slide, 8.95, 5.1, 3.95, 1.7, insight_text, 13)

    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    add_textbox(slide, 0.45, 0.18, 12.3, 0.45, "六大场景统计：学段与学科分布", 24, True)
    add_textbox(slide, 0.48, 0.62, 12.0, 0.35, "左侧为学段×场景热力图，右侧为 Top10 学科×场景热力图", 11)
    slide.shapes.add_picture(str(figures["stage_scene"]), PptInches(0.45), PptInches(1.0), width=PptInches(6.2))
    slide.shapes.add_picture(str(figures["subject_scene"]), PptInches(6.85), PptInches(1.0), width=PptInches(5.95))
    summary_lines = [
        f"- {scene}最集中学段：{dominant_stage.get(scene, '未提及')}；最集中学科：{dominant_subject.get(scene, '未提及')}"
        for scene in SCENE_ORDER
    ]
    add_textbox(slide, 0.55, 6.35, 12.0, 0.8, "\n".join(summary_lines[:3]), 10)
    add_textbox(slide, 6.85, 6.35, 5.95, 0.8, "\n".join(summary_lines[3:]), 10)

    presentation.save(PPTX_PATH)


def main() -> None:
    configure_matplotlib()
    bundle = load_bundle()
    save_workbook_summary(bundle)
    figures = generate_figures(bundle)
    build_docx(bundle, figures)
    build_ppt(bundle, figures)
    print(f"Generated docx: {DOCX_PATH}")
    print(f"Generated pptx: {PPTX_PATH}")
    print(f"Workbook summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()