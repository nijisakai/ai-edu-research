#!/usr/bin/env python3
"""
Render consulting-style research report Markdown to premium PDF.
Uses weasyprint for HTML->PDF conversion with modern consulting styling.
"""

import re
import os
import sys
import markdown
from pathlib import Path

# Paths
REPORT_MD = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/report/研究报告_AI赋能基础教育实践图景与产业生态分析.md")
REPORT_HTML = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/report/研究报告_AI赋能基础教育实践图景与产业生态分析.html")
REPORT_PDF = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/report/研究报告_AI赋能基础教育实践图景与产业生态分析.pdf")
FIGURES_DIR = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/figures")

# Color scheme
TEAL = "#0D4F4F"
TEAL_LIGHT = "#1A7A7A"
TEAL_PALE = "#E8F4F4"
GOLD = "#C19A3E"
GOLD_LIGHT = "#D4B76A"
GOLD_PALE = "#FDF6E8"


def read_markdown():
    """Read the report markdown file."""
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        return f.read()


def get_figure_files():
    """Get all PNG figure files sorted."""
    if FIGURES_DIR.exists():
        return sorted(FIGURES_DIR.glob("*.png"))
    return []


def parse_report_structure(md_text):
    """Parse markdown into structured sections."""
    # Title
    title_match = re.match(r'^#\s+(.+)', md_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "研究报告"

    # Subtitle
    subtitle_match = re.search(r'^##\s+(——.+)', md_text, re.MULTILINE)
    subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

    # Extract key stats from the report metadata
    stats = {
        "cases": "1,690",
        "tools": "1,830",
        "companies": "1,726",
        "provinces": "30",
    }

    # Extract summary/abstract
    abstract_match = re.search(r'##\s*摘要\s*\n\n(.+?)(?=\n---|\n##)', md_text, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    # Extract sections - find all ## headings
    sections = []
    section_pattern = re.compile(r'^##\s+(.+?)$', re.MULTILINE)
    matches = list(section_pattern.finditer(md_text))

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        # Skip subtitle, abstract
        if heading.startswith('——') or heading == '摘要':
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()

        # Clean up separators
        content = re.sub(r'^---\s*$', '', content, flags=re.MULTILINE).strip()

        sections.append({
            "heading": heading,
            "content": content,
        })

    return {
        "title": title,
        "subtitle": subtitle,
        "stats": stats,
        "abstract": abstract,
        "sections": sections,
    }


def md_to_html(md_text):
    """Convert markdown to HTML."""
    extensions = ['tables', 'footnotes', 'toc', 'smarty']
    html = markdown.markdown(md_text, extensions=extensions)
    return html


def process_report_html(html):
    """Post-process HTML for report styling."""
    # Add table class
    html = html.replace('<table>', '<table class="report-table">')

    # Style blockquotes as callout boxes
    # Detect blockquotes with case studies
    html = re.sub(
        r'<blockquote>\s*<p>\s*<strong>案例聚焦',
        '<blockquote class="case-callout"><p><strong>案例聚焦',
        html
    )

    # Style other blockquotes
    html = html.replace('<blockquote>', '<blockquote class="callout-box">')

    # Fix figure references - embed actual figure images
    figure_files = get_figure_files()
    fig_map = {}
    for f in figure_files:
        fig_map[f.stem] = f

    # Replace (见图X) references with actual images where possible
    def insert_figure(match):
        fig_text = match.group(0)
        # Try to find a matching figure number
        fig_num_match = re.search(r'fig(\d+)', fig_text, re.IGNORECASE)
        if fig_num_match:
            num = fig_num_match.group(1)
            for stem, path in fig_map.items():
                if f"fig{num}" in stem or f"fig{num.zfill(2)}" in stem:
                    return f'{fig_text}<div class="figure-embed"><img src="file://{path}" alt="{stem}"></div>'
        return fig_text

    return html


def build_toc_html(sections):
    """Build table of contents."""
    toc = '<div class="toc-page">\n'
    toc += '<h2 class="toc-title">目　录</h2>\n'
    toc += '<div class="toc-entries">\n'

    for i, section in enumerate(sections):
        heading = section["heading"]
        # Determine level
        level = "toc-l1"
        # Assign number for visual hierarchy
        toc += f'<div class="{level}"><span class="toc-text">{heading}</span></div>\n'

    toc += '</div>\n</div>\n'
    return toc


def build_section_html(section, fig_map, section_index):
    """Build HTML for a single section with divider."""
    heading = section["heading"]
    content_html = md_to_html(section["content"])
    content_html = process_report_html(content_html)

    # Embed relevant figures inline
    content_html = embed_figures_in_content(content_html, fig_map)

    html = ""

    # Add section divider for major sections
    major_sections = ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、"]
    is_major = any(heading.startswith(s) for s in major_sections)

    if is_major:
        # Extract section number for divider
        sec_num = heading.split("、")[0] if "、" in heading else str(section_index + 1)
        sec_title = heading.split("、")[1] if "、" in heading else heading
        html += f'''
<div class="section-divider">
    <div class="section-divider-content">
        <div class="section-divider-num">{sec_num}</div>
        <div class="section-divider-title">{sec_title if "、" in heading else heading}</div>
    </div>
</div>
'''

    html += f'<div class="section-content">\n'
    html += f'<h2 class="section-heading">{heading}</h2>\n'
    html += content_html
    html += '</div>\n'

    return html


def embed_figures_in_content(html, fig_map):
    """Find figure references and embed images."""
    # Pattern: (见图X) or （见图X） or 图X
    def replace_fig_ref(match):
        fig_label = match.group(0)
        # Extract figure number - look for numbers
        num_match = re.search(r'(\d+)', fig_label)
        if num_match:
            num = num_match.group(1)
            padded = num.zfill(2)
            # Search for matching figure file
            for stem, path in fig_map.items():
                if f"fig{padded}_" in stem or f"fig{num}_" in stem:
                    img_html = f'<div class="figure-embed"><img src="file://{path}" alt="{stem}"><p class="figure-caption-text">{fig_label}</p></div>'
                    return img_html
        return fig_label

    # Match patterns like (见图1), （见图10）, etc.
    html = re.sub(r'[（(]见图\d+[）)]', replace_fig_ref, html)

    return html


def build_report_html(report):
    """Build the complete report HTML."""
    fig_files = get_figure_files()
    fig_map = {f.stem: f for f in fig_files}

    # Key figure files for the cover
    key_figures = []
    for name in ['fig01_province_heatmap', 'fig04_product_ecosystem_treemap',
                  'fig05_scenario_sunburst', 'fig18_comprehensive_dashboard']:
        for stem, path in fig_map.items():
            if name in stem:
                key_figures.append(path)
                break

    sections_html = ""
    for i, section in enumerate(report["sections"]):
        sections_html += build_section_html(section, fig_map, i)

    # Build gallery of all key figures
    figures_gallery = ""
    important_figs = [
        ("fig01_province_heatmap", "省域分布热力图"),
        ("fig02_school_stage_donut", "学段分布"),
        ("fig03_top20_tools_bar", "工具使用Top20"),
        ("fig04_product_ecosystem_treemap", "产品生态树图"),
        ("fig05_scenario_sunburst", "应用场景旭日图"),
        ("fig06_subject_distribution", "学科分布"),
        ("fig07_cultivation_radar", "五育雷达图"),
        ("fig10_stage_subject_heatmap", "学段-学科热力图"),
        ("fig11_industry_maturity", "产业成熟度"),
        ("fig13_company_market_concentration", "市场集中度"),
        ("fig14_lda_topics", "LDA主题模型"),
        ("fig15_tech_pathway_sankey", "技术路径桑基图"),
        ("fig18_comprehensive_dashboard", "综合仪表板"),
        ("fig_f1_sanfuneng_istar_heatmap", "三赋能/iSTAR热力图"),
        ("fig_f3_digital_pedagogy_radar", "数字教学法雷达图"),
        ("fig_f4_smart_edu_sankey", "智慧教育桑基图"),
        ("fig_f5_innovation_depth", "创新深度分析"),
        ("fig_f7_framework_dashboard", "理论框架仪表板"),
        ("fig_s3_correlation_matrix", "相关矩阵"),
        ("fig_s9_effect_size_forest", "效应量森林图"),
    ]

    for fig_stem, fig_title in important_figs:
        for stem, path in fig_map.items():
            if fig_stem in stem:
                figures_gallery += f'''
<div class="gallery-item">
    <img src="file://{path}" alt="{fig_title}">
    <p class="gallery-caption">{fig_title}</p>
</div>
'''
                break

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{report['title']}</title>
<style>
/* ============================================
   CONSULTING REPORT STYLESHEET
   Premium design with teal/gold color scheme
   ============================================ */

/* Page setup */
@page {{
    size: A4;
    margin: 2cm 2cm 2.5cm 2cm;

    @top-left {{
        content: "AI赋能基础教育研究报告";
        font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
        font-size: 8pt;
        color: {TEAL};
        opacity: 0.7;
    }}

    @top-right {{
        content: "2026";
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 8pt;
        color: {GOLD};
        opacity: 0.7;
    }}

    @bottom-center {{
        content: counter(page);
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 9pt;
        color: {TEAL};
    }}

    @bottom-left {{
        content: "";
        border-top: 0.5pt solid {TEAL};
        width: 100%;
        opacity: 0.3;
    }}
}}

@page:first {{
    margin: 0;
    @top-left {{ content: none; }}
    @top-right {{ content: none; }}
    @bottom-center {{ content: none; }}
    @bottom-left {{ content: none; }}
}}

/* Base typography */
body {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #2a2a2a;
    text-align: justify;
}}

/* ---- COVER PAGE ---- */
.cover-page {{
    page-break-after: always;
    width: 210mm;
    height: 297mm;
    position: relative;
    overflow: hidden;
    margin: -2cm;
    padding: 0;
    background: linear-gradient(135deg, {TEAL} 0%, #0A3D3D 40%, #072E2E 70%, #051F1F 100%);
    color: white;
}}

.cover-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(180deg, rgba(193,154,62,0.08) 0%, rgba(193,154,62,0.02) 50%, rgba(0,0,0,0.3) 100%);
}}

.cover-content {{
    position: relative;
    z-index: 1;
    padding: 4.5cm 3cm 3cm 3cm;
}}

.cover-tag {{
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 9pt;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {GOLD};
    margin-bottom: 1cm;
    font-weight: 500;
}}

.cover-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 30pt;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 0.5cm;
    color: #ffffff;
    letter-spacing: 0.02em;
}}

.cover-subtitle {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 16pt;
    font-weight: 300;
    color: rgba(255,255,255,0.8);
    margin-bottom: 2cm;
    letter-spacing: 0.05em;
}}

.cover-divider {{
    width: 60pt;
    height: 2pt;
    background: {GOLD};
    margin-bottom: 1.5cm;
}}

.cover-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    margin-bottom: 2cm;
}}

.cover-stat {{
    width: 48%;
    margin-bottom: 0.6cm;
}}

.cover-stat-value {{
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 28pt;
    font-weight: 700;
    color: {GOLD};
    line-height: 1.2;
}}

.cover-stat-label {{
    font-size: 9pt;
    color: rgba(255,255,255,0.65);
    letter-spacing: 0.05em;
}}

.cover-date {{
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 10pt;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.1em;
    position: absolute;
    bottom: 3cm;
    left: 3cm;
}}

.cover-accent-line {{
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 4pt;
    background: linear-gradient(90deg, {GOLD} 0%, {GOLD_LIGHT} 50%, transparent 100%);
}}

/* ---- TABLE OF CONTENTS ---- */
.toc-page {{
    page-break-after: always;
    padding-top: 2cm;
}}

.toc-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 22pt;
    font-weight: 700;
    color: {TEAL};
    margin-bottom: 1.5cm;
    padding-bottom: 0.3cm;
    border-bottom: 2pt solid {GOLD};
    text-indent: 0;
}}

.toc-entries {{
    margin-top: 0.5cm;
}}

.toc-l1 {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: {TEAL};
    padding: 0.25cm 0;
    border-bottom: 0.5pt solid #eee;
}}

.toc-text {{
    text-indent: 0;
}}

/* ---- SECTION DIVIDERS ---- */
.section-divider {{
    page-break-before: always;
    page-break-after: avoid;
    width: calc(100% + 4cm);
    margin-left: -2cm;
    margin-right: -2cm;
    margin-top: -2cm;
    padding: 4cm 3cm 3cm 3cm;
    background: linear-gradient(135deg, {TEAL} 0%, {TEAL_LIGHT} 100%);
    color: white;
    min-height: 6cm;
}}

.section-divider-content {{
    max-width: 80%;
}}

.section-divider-num {{
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 48pt;
    font-weight: 200;
    color: {GOLD};
    line-height: 1;
    margin-bottom: 0.5cm;
    opacity: 0.9;
}}

.section-divider-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 22pt;
    font-weight: 700;
    line-height: 1.3;
    color: white;
}}

/* ---- SECTION CONTENT ---- */
.section-content {{
    margin-bottom: 1cm;
}}

/* ---- HEADINGS ---- */
h2.section-heading {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: {TEAL};
    margin-top: 1.2cm;
    margin-bottom: 0.5cm;
    padding-bottom: 0.2cm;
    border-bottom: 1.5pt solid {GOLD};
    line-height: 1.4;
    page-break-after: avoid;
    text-indent: 0;
}}

h3 {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 14pt;
    font-weight: 700;
    color: {TEAL};
    margin-top: 0.8cm;
    margin-bottom: 0.4cm;
    line-height: 1.4;
    page-break-after: avoid;
    text-indent: 0;
}}

h4 {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 12pt;
    font-weight: 700;
    color: {TEAL_LIGHT};
    margin-top: 0.6cm;
    margin-bottom: 0.3cm;
    line-height: 1.4;
    page-break-after: avoid;
    text-indent: 0;
}}

/* ---- BODY TEXT ---- */
p {{
    margin-bottom: 0.3cm;
    line-height: 1.7;
    text-indent: 2em;
}}

/* Bold text */
strong {{
    font-weight: 700;
    color: #1a1a1a;
}}

/* ---- TABLES ---- */
.report-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.6cm 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}

.report-table thead {{
    background: {TEAL};
    color: white;
}}

.report-table thead th {{
    padding: 8pt 10pt;
    text-align: left;
    font-weight: 600;
    font-size: 10pt;
    letter-spacing: 0.02em;
    text-indent: 0;
}}

.report-table tbody td {{
    padding: 7pt 10pt;
    border-bottom: 0.5pt solid #e5e5e5;
    font-size: 10pt;
    text-indent: 0;
}}

.report-table tbody tr:nth-child(even) {{
    background-color: {TEAL_PALE};
}}

.report-table tbody tr:hover {{
    background-color: #f0f8f8;
}}

/* ---- CALLOUT BOXES ---- */
.callout-box {{
    margin: 0.6cm 0;
    padding: 0.6cm 0.8cm;
    border-left: 4pt solid {GOLD};
    background: {GOLD_PALE};
    border-radius: 0 4pt 4pt 0;
    font-size: 10.5pt;
    line-height: 1.6;
    page-break-inside: avoid;
}}

.callout-box p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
    color: #333;
}}

.callout-box strong {{
    color: {TEAL};
}}

.case-callout {{
    margin: 0.6cm 0;
    padding: 0.6cm 0.8cm;
    border-left: 4pt solid {TEAL};
    background: {TEAL_PALE};
    border-radius: 0 4pt 4pt 0;
    font-size: 10.5pt;
    line-height: 1.6;
    page-break-inside: avoid;
}}

.case-callout p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
}}

.case-callout strong {{
    color: {TEAL};
}}

/* ---- FIGURES ---- */
.figure-embed {{
    margin: 0.6cm 0;
    text-align: center;
    page-break-inside: avoid;
}}

.figure-embed img {{
    max-width: 100%;
    height: auto;
    max-height: 16cm;
    border: 0.5pt solid #e5e5e5;
    border-radius: 3pt;
    box-shadow: 0 1pt 3pt rgba(0,0,0,0.08);
}}

.figure-caption-text {{
    font-size: 9pt;
    color: #666;
    text-align: center;
    margin-top: 0.2cm;
    text-indent: 0;
}}

/* ---- FIGURE GALLERY ---- */
.figures-page {{
    page-break-before: always;
}}

.figures-page-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: {TEAL};
    margin-bottom: 0.8cm;
    padding-bottom: 0.2cm;
    border-bottom: 1.5pt solid {GOLD};
    text-indent: 0;
}}

.gallery-item {{
    margin-bottom: 0.4cm;
    page-break-inside: avoid;
    text-align: center;
}}

.gallery-item img {{
    max-width: 100%;
    height: auto;
    max-height: 14cm;
    border: 0.5pt solid #e5e5e5;
    border-radius: 3pt;
}}

.gallery-caption {{
    font-size: 9.5pt;
    color: {TEAL};
    font-weight: 600;
    text-align: center;
    margin-top: 0.2cm;
    margin-bottom: 0.4cm;
    text-indent: 0;
}}

/* ---- LISTS ---- */
ul, ol {{
    margin: 0.3cm 0 0.3cm 1.5cm;
    padding: 0;
}}

li {{
    margin-bottom: 0.15cm;
    line-height: 1.6;
    text-indent: 0;
}}

li p {{
    text-indent: 0;
    margin-bottom: 0.1cm;
}}

/* ---- HORIZONTAL RULES ---- */
hr {{
    border: none;
    border-top: 1pt solid {GOLD};
    margin: 1cm 0;
    opacity: 0.4;
}}

/* ---- FOOTNOTES ---- */
.footnote {{
    font-size: 9pt;
    color: #666;
    line-height: 1.4;
}}

.footnote-ref {{
    font-size: 8pt;
    vertical-align: super;
    color: {TEAL};
}}

/* ---- ABSTRACT BOX ---- */
.abstract-box {{
    margin: 1cm 0;
    padding: 0.8cm 1cm;
    background: linear-gradient(135deg, {TEAL_PALE} 0%, #f5fafa 100%);
    border: 1pt solid #c5dede;
    border-radius: 4pt;
    font-size: 10.5pt;
    line-height: 1.6;
}}

.abstract-box p {{
    text-indent: 2em;
    margin-bottom: 0.2cm;
    color: #333;
}}

.abstract-box-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 13pt;
    font-weight: 700;
    color: {TEAL};
    margin-bottom: 0.4cm;
    text-indent: 0;
}}

/* Avoid page break issues */
h2, h3, h4 {{
    page-break-after: avoid;
}}

.report-table, .callout-box, .case-callout {{
    page-break-inside: avoid;
}}

/* ---- ENDMATTER ---- */
.endmatter {{
    margin-top: 1cm;
    padding-top: 0.5cm;
    border-top: 0.5pt solid #ddd;
    font-size: 9pt;
    color: #888;
}}

.endmatter p {{
    text-indent: 0;
    margin-bottom: 0.15cm;
}}

</style>
</head>
<body>

<!-- ======== COVER PAGE ======== -->
<div class="cover-page">
    <div class="cover-overlay"></div>
    <div class="cover-content">
        <div class="cover-tag">RESEARCH REPORT &middot; 研究报告</div>
        <h1 class="cover-title">{report['title']}</h1>
        <div class="cover-subtitle">{report['subtitle']}</div>
        <div class="cover-divider"></div>
        <div class="cover-stats">
            <div class="cover-stat">
                <div class="cover-stat-value">{report['stats']['cases']}</div>
                <div class="cover-stat-label">有效案例</div>
            </div>
            <div class="cover-stat">
                <div class="cover-stat-value">{report['stats']['tools']}</div>
                <div class="cover-stat-label">独立工具/产品</div>
            </div>
            <div class="cover-stat">
                <div class="cover-stat-value">{report['stats']['companies']}</div>
                <div class="cover-stat-label">企业/机构主体</div>
            </div>
            <div class="cover-stat">
                <div class="cover-stat-value">{report['stats']['provinces']}</div>
                <div class="cover-stat-label">省级行政区覆盖</div>
            </div>
        </div>
        <div class="cover-date">2026年2月</div>
    </div>
    <div class="cover-accent-line"></div>
</div>

<!-- ======== TABLE OF CONTENTS ======== -->
{build_toc_html(report['sections'])}

<!-- ======== ABSTRACT ======== -->
<div class="abstract-box">
    <div class="abstract-box-title">摘　要</div>
    <p>{report['abstract']}</p>
</div>

<!-- ======== SECTIONS ======== -->
{sections_html}

<!-- ======== FIGURE GALLERY ======== -->
<div class="figures-page">
    <h2 class="figures-page-title">附录：核心图表</h2>
    {figures_gallery}
</div>

<!-- ======== ENDMATTER ======== -->
<div class="endmatter">
    <p>本报告基于1690个AI教育应用案例的数据分析生成。报告中的分析和建议仅代表研究团队基于数据的学术判断。</p>
    <p>数据截止至案例征集活动结束日期。 &copy; 2026</p>
</div>

</body>
</html>
"""
    return html


def main():
    print("=" * 60)
    print("Consulting Report PDF Renderer")
    print("=" * 60)

    # Read markdown
    print(f"\n[1/4] Reading markdown: {REPORT_MD}")
    md_text = read_markdown()
    print(f"      Read {len(md_text)} characters")

    # Parse structure
    print("[2/4] Parsing report structure...")
    report = parse_report_structure(md_text)
    print(f"      Title: {report['title']}")
    print(f"      Sections: {len(report['sections'])}")
    print(f"      Abstract: {len(report['abstract'])} chars")

    # Check figures
    fig_files = get_figure_files()
    print(f"      Figures found: {len(fig_files)}")

    # Build HTML
    print("[3/4] Building HTML...")
    html = build_report_html(report)

    # Save HTML
    print(f"      Saving HTML: {REPORT_HTML}")
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"      HTML size: {os.path.getsize(REPORT_HTML) / 1024:.1f} KB")

    # Convert to PDF
    print(f"[4/4] Converting to PDF: {REPORT_PDF}")
    print("      (This may take a while due to embedded figures...)")
    try:
        import weasyprint
        from weasyprint import HTML
        from weasyprint.text.fonts import FontConfiguration

        font_config = FontConfiguration()
        doc = HTML(string=html, base_url=str(REPORT_HTML.parent))
        doc.write_pdf(
            str(REPORT_PDF),
            font_config=font_config,
        )
        pdf_size = os.path.getsize(REPORT_PDF)
        print(f"      PDF size: {pdf_size / 1024:.1f} KB ({pdf_size / (1024*1024):.2f} MB)")
        print("\n[DONE] Report PDF generated successfully!")
    except Exception as e:
        print(f"\n[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
