#!/usr/bin/env python3
"""
Comprehensive rebuild script for paper and report.
Embeds all high-quality figures (fig_a01-fig_f06) into both documents.
Generates HTML and PDF outputs.
"""

import re
import os
import sys
import markdown
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE = Path("/Users/sakai/Desktop/产业调研/ai-edu-research")
FIGURES_DIR = BASE / "output" / "figures"

PAPER_MD = BASE / "output" / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.md"
PAPER_HTML = BASE / "output" / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.html"
PAPER_PDF = BASE / "output" / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.pdf"

REPORT_MD = BASE / "output" / "report" / "研究报告_AI赋能基础教育实践图景与产业生态分析.md"
REPORT_HTML = BASE / "output" / "report" / "研究报告_AI赋能基础教育实践图景与产业生态分析.html"
REPORT_PDF = BASE / "output" / "report" / "研究报告_AI赋能基础教育实践图景与产业生态分析.pdf"

# Absolute path prefix for figures (WeasyPrint needs this)
FIG_ABS = str(FIGURES_DIR)

# =============================================================================
# FIGURE DEFINITIONS — mapping new high-quality figures to semantic positions
# =============================================================================
# Paper figure numbering (学术论文)
PAPER_FIGURES = {
    "fig1": ("fig_a01_province_map.png", "图1 1690个AI教育案例的省域分布热力图"),
    "fig2": ("fig_a02_stage_waffle.png", "图2 AI教育案例的学段分布（华夫饼图）"),
    "fig3": ("fig_a03_subject_lollipop.png", "图3 各学科AI应用渗透率（棒棒糖图）"),
    "fig4": ("fig_a04_scenario_treemap.png", "图4 四类应用场景的案例体量对比（树图）"),
    "fig5": ("fig_a05_five_edu_radar.png", "图5 五育维度的案例数量与人机协同深度（雷达图）"),
    "fig6": ("fig_b01_top20_tools.png", "图6 AI教育工具使用频次Top20（水平条形图）"),
    "fig7": ("fig_b02_tech_sankey.png", "图7 技术要素-应用场景-教学环节的多级流转路径（桑基图）"),
    "fig8": ("fig_b03_model_landscape.png", "图8 AI教育大模型产品的市场定位与覆盖面（景观图）"),
    "fig9": ("fig_b04_cooccurrence_network.png", "图9 AI工具共现关系网络图"),
    "fig10": ("fig_c01_umap_clusters.png", "图10 1690个案例的UMAP降维聚类可视化"),
    "fig11": ("fig_d01_lorenz_curve.png", "图11 AI教育工具市场集中度洛伦兹曲线"),
    "fig12": ("fig_d02_ecosystem_treemap.png", "图12 EdTech产品生态的层次结构（嵌套树图）"),
    "fig13": ("fig_f05_geographic_inequality.png", "图13 AI教育创新深度的地理不平等Theil指数分解"),
    "fig14": ("fig_e01_sanfuneng_istar.png", "图14 三赋能框架与iSTAR人机协同层级映射"),
    "fig15": ("fig_e03_three_realms.png", "图15 智慧教育三重境界的递进关系与当前阶段定位"),
    "fig16": ("fig_e04_digital_pedagogy_radar.png", "图16 数字教学法四维框架达成度诊断（雷达图）"),
    "fig17": ("fig_e02_innovation_ridgeline.png", "图17 10个聚类的创新深度评分分布（岭线图）"),
    "fig18": ("fig_e05_techgen_scenario_bubble.png", "图18 技术代际与应用场景的交叉分布（气泡图）"),
    "fig19": ("fig_f01_rf_shap.png", "图19 随机森林模型SHAP特征重要性（蜂群图）"),
    "fig20": ("fig_f02_cramers_v.png", "图20 关键变量间Cramér's V关联强度（热力图）"),
    "fig21": ("fig_f03_correspondence_biplot.png", "图21 学段-学科-场景对应分析（双标图）"),
    "fig22": ("fig_f04_mca_biplot.png", "图22 多维特征联合关系的多重对应分析（MCA双标图）"),
    "fig23": ("fig_f06_cluster_anova.png", "图23 聚类方差分析效应量对比（柱状图）"),
    "fig24": ("fig_e06_dashboard.png", "图24 研究核心数据指标综合仪表板"),
}

# Report figure mapping
REPORT_FIGURES = {
    # Section 2: 教育视角
    "fig_edu_province": ("fig_a01_province_map.png", "图1 案例省域分布地图"),
    "fig_edu_stage": ("fig_a02_stage_waffle.png", "图2 学段分布华夫饼图"),
    "fig_edu_subject": ("fig_a03_subject_lollipop.png", "图3 学科渗透棒棒糖图"),
    "fig_edu_scenario": ("fig_a04_scenario_treemap.png", "图4 应用场景树图"),
    "fig_edu_wuyu": ("fig_a05_five_edu_radar.png", "图5 五育融合雷达图"),
    "fig_edu_cluster": ("fig_c01_umap_clusters.png", "图6 案例聚类UMAP可视化"),
    # Section 3: 技术视角
    "fig_tech_tools": ("fig_b01_top20_tools.png", "图7 AI工具使用频次Top20"),
    "fig_tech_sankey": ("fig_b02_tech_sankey.png", "图8 技术路径桑基图"),
    "fig_tech_model": ("fig_b03_model_landscape.png", "图9 AI大模型景观图"),
    "fig_tech_network": ("fig_b04_cooccurrence_network.png", "图10 技术共现网络"),
    # Section 4: 产业视角
    "fig_ind_lorenz": ("fig_d01_lorenz_curve.png", "图11 市场集中度洛伦兹曲线"),
    "fig_ind_ecosystem": ("fig_d02_ecosystem_treemap.png", "图12 EdTech产品生态树图"),
    # Section 5: 区域分析
    "fig_geo_inequality": ("fig_f05_geographic_inequality.png", "图13 地理不平等分析"),
    # Section 6: 理论框架
    "fig_fw_istar": ("fig_e01_sanfuneng_istar.png", "图14 三赋能与iSTAR框架映射"),
    "fig_fw_three_realms": ("fig_e03_three_realms.png", "图15 智慧教育三重境界"),
    "fig_fw_pedagogy": ("fig_e04_digital_pedagogy_radar.png", "图16 数字教学法雷达图"),
    "fig_fw_innovation": ("fig_e02_innovation_ridgeline.png", "图17 创新深度岭线图"),
    "fig_fw_bubble": ("fig_e05_techgen_scenario_bubble.png", "图18 技术代际-场景气泡图"),
    # Section 7.5: 深度洞察
    "fig_deep_shap": ("fig_f01_rf_shap.png", "图19 SHAP特征重要性分析"),
    "fig_deep_cramers": ("fig_f02_cramers_v.png", "图20 Cramér's V关联热力图"),
    "fig_deep_ca": ("fig_f03_correspondence_biplot.png", "图21 对应分析双标图"),
    "fig_deep_mca": ("fig_f04_mca_biplot.png", "图22 多重对应分析双标图"),
    "fig_deep_anova": ("fig_f06_cluster_anova.png", "图23 聚类方差分析"),
    "fig_deep_dashboard": ("fig_e06_dashboard.png", "图24 研究综合仪表板"),
}


def fig_html(filename, caption, for_paper=True):
    """Generate HTML for an embedded figure."""
    path = f"{FIG_ABS}/{filename}"
    if not os.path.exists(path):
        print(f"  [WARN] Figure not found: {path}")
        return ""

    if for_paper:
        return f'''
<div class="figure-block">
    <img src="file://{path}" alt="{caption}">
    <p class="fig-caption"><strong>{caption}</strong></p>
</div>
'''
    else:
        return f'''
<div class="figure-embed">
    <img src="file://{path}" alt="{caption}">
    <p class="figure-caption-text"><strong>{caption}</strong></p>
</div>
'''


# =============================================================================
# PAPER BUILDER
# =============================================================================
def build_paper():
    """Build the paper HTML with all figures embedded."""
    print("\n" + "=" * 60)
    print("REBUILDING ACADEMIC PAPER")
    print("=" * 60)

    md_text = PAPER_MD.read_text(encoding="utf-8")
    print(f"  Read {len(md_text)} chars from MD")

    # Parse paper structure
    paper = parse_paper_structure(md_text)

    # Convert body to HTML
    body_html = md_to_html(paper["body_md"])
    body_html = post_process_paper_html(body_html)

    # Insert figures at strategic positions
    body_html = insert_paper_figures(body_html)

    # Process references
    refs_html = process_references(paper["refs_md"])

    # Process endmatter
    endmatter_html = md_to_html(paper["endmatter_md"]) if paper["endmatter_md"] else ""

    # Build full HTML
    html = PAPER_HTML_TEMPLATE.format(
        title=paper["title"],
        subtitle=paper["subtitle"],
        cn_abstract=paper["cn_abstract"],
        cn_keywords=paper["cn_keywords"],
        en_abstract=paper["en_abstract"],
        en_keywords=paper["en_keywords"],
        body_html=body_html,
        refs_html=refs_html,
        endmatter_html=endmatter_html,
    )

    # Save HTML
    PAPER_HTML.write_text(html, encoding="utf-8")
    print(f"  Paper HTML saved: {os.path.getsize(PAPER_HTML) / 1024:.1f} KB")

    # Render PDF
    render_pdf(html, PAPER_PDF, str(PAPER_HTML.parent))


def parse_paper_structure(md_text):
    """Parse paper markdown into structured components."""
    title_match = re.match(r'^#\s+(.+)', md_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "论文"

    subtitle_match = re.search(r'^##\s+(——.+)', md_text, re.MULTILINE)
    subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

    cn_abstract_match = re.search(
        r'\*\*摘要\*\*[：:]\s*(.+?)(?=\n\n\*\*关键词\*\*)', md_text, re.DOTALL)
    cn_abstract = cn_abstract_match.group(1).strip() if cn_abstract_match else ""

    cn_keywords_match = re.search(
        r'\*\*关键词\*\*[：:]\s*(.+?)(?=\n\n)', md_text, re.DOTALL)
    cn_keywords = cn_keywords_match.group(1).strip() if cn_keywords_match else ""

    en_abstract_match = re.search(
        r'\*\*Abstract\*\*[：:]\s*(.+?)(?=\n\n\*\*Keywords\*\*)', md_text, re.DOTALL)
    en_abstract = en_abstract_match.group(1).strip() if en_abstract_match else ""

    en_keywords_match = re.search(
        r'\*\*Keywords\*\*[：:]\s*(.+?)(?=\n\n)', md_text, re.DOTALL)
    en_keywords = en_keywords_match.group(1).strip() if en_keywords_match else ""

    body_start = md_text.find("## 一、引言")
    refs_start = md_text.find("## 参考文献")

    body_md = ""
    refs_md = ""
    endmatter_md = ""

    if body_start > 0 and refs_start > 0:
        body_md = md_text[body_start:refs_start].strip()
        after_refs = md_text[refs_start:]
        endmatter_match = re.search(r'\n---\n', after_refs)
        if endmatter_match:
            refs_md = after_refs[:endmatter_match.start()].strip()
            endmatter_md = after_refs[endmatter_match.end():].strip()
        else:
            refs_md = after_refs.strip()
    elif body_start > 0:
        body_md = md_text[body_start:].strip()

    return {
        "title": title,
        "subtitle": subtitle,
        "cn_abstract": cn_abstract,
        "cn_keywords": cn_keywords,
        "en_abstract": en_abstract,
        "en_keywords": en_keywords,
        "body_md": body_md,
        "refs_md": refs_md,
        "endmatter_md": endmatter_md,
    }


def md_to_html(md_text):
    """Convert markdown to HTML."""
    if not md_text:
        return ""
    extensions = ['tables', 'footnotes', 'toc', 'smarty']
    return markdown.markdown(md_text, extensions=extensions)


def post_process_paper_html(html):
    """Post-process paper HTML for academic styling."""
    # Add three-line table class
    html = html.replace('<table>', '<table class="three-line-table">')

    # Convert blockquote tables/figures to proper containers
    html = re.sub(
        r'<blockquote>\s*<p>\s*<strong>(表\d+[^<]*)</strong>\s*</p>',
        r'<div class="table-container"><p class="table-caption"><strong>\1</strong></p>',
        html
    )
    html = re.sub(
        r'<blockquote>\s*<p>\s*<strong>(图\d+[^<]*)</strong>\s*</p>',
        r'<div class="figure-caption"><p class="caption-text"><strong>\1</strong></p>',
        html
    )
    html = html.replace('</blockquote>', '</div>')

    return html


def insert_paper_figures(html):
    """Insert figure images at strategic positions in paper HTML."""

    # Define insertion rules: (marker_text, figure_key)
    # We insert the figure AFTER the element containing the marker
    insertions = [
        # Section 3.3 — analysis framework
        ("三个维度相互关联、彼此支撑", "fig14"),
        # Section 4.1.1 — province distribution
        ("浙江省以432个案例位居首位", "fig1"),
        # After stage table — stage waffle
        ("幼儿园阶段的AI应用占比达到11.2%", "fig2"),
        # Subject — after discussion of subject penetration
        ("AI在图像生成", None),  # skip, subject comes after scenario
        # Section 4.1.2 — scenario
        ("助学&rdquo;类场景占据绝对主导地位（73.5%）", "fig4"),
        # Section 4.1.3 — wuyu
        ("人机协同深度反而更高", "fig5"),
        # Section 4.1.4 — cluster
        ("从更细粒度上揭示了AI教学应用的多样化模式", "fig10"),
        ("聚类结果具有良好的结构效度", "fig23"),
        # Section 4.2.1 — tools
        ("生成式AI已成为当前AI教育应用的", "fig6"),
        # Section 4.2.2 — tech pathway
        ("许多案例仅涉及路径中的部分环节", "fig7"),
        # Section 4.2.3 — product form
        ("正在成为AI教育产品的主流形态", "fig8"),
        # Section 4.2.4 — tool combo
        ("技术整合能力&rdquo;而非&rdquo;单一工具使用", "fig9"),
        # Section 4.3.1 — market
        ("进一步印证了市场的高度分散性", "fig11"),
        ("可能带来数据垄断", "fig12"),
        # Subject figure — insert after lollipop reference point
        ("美术教学的异军突起", None),  # skip
        # Section 4.3.4 — regional
        ("省份内部、学校之间乃至教师个体之间的微观差距", "fig13"),
        # Section 5.1 — scene generalization
        ("帮助教师从被动的技术使用者转变为主动的场景创新者", "fig3"),
        # Section 5.2 — digital pedagogy
        ("诊断为未来AI教育技术的发展方向提供了明确的指引", "fig16"),
        # Section 5.3 — dual empowerment
        ("促进创新与防范风险之间寻求动态平衡", "fig18"),
        # Section 5.4 — iSTAR
        ("先行者群体的实践经验为大规模推动", "fig17"),
        # Section 5.5 — three realms
        ("这一目标的实现仍需要长期的、多层面的协同努力", "fig15"),
        # Section 5.6 — counterintuitive
        ("教师将技术转化为教育价值的专业能力", "fig20"),
        ("技术的&rdquo;新&rdquo;不等于教学的&rdquo;深", "fig19"),
        # MCA and CA - insert after SHAP analysis paragraph
        ("Gini重要性与SHAP值的排序高度一致", "fig21"),
        ("序数Logistic回归的结果与此呼应", "fig22"),
        # Section 6 conclusion — dashboard
        ("为理论的迭代完善提供了方向", "fig24"),
    ]

    for marker, fig_key in insertions:
        if fig_key is None:
            continue
        if fig_key not in PAPER_FIGURES:
            continue

        filename, caption = PAPER_FIGURES[fig_key]
        fig_block = fig_html(filename, caption, for_paper=True)

        if not fig_block:
            continue

        # Find the paragraph containing the marker and insert figure after it
        pos = html.find(marker)
        if pos < 0:
            # Try without HTML entities
            clean_marker = marker.replace("&rdquo;", "").replace("&ldquo;", "")
            pos = html.find(clean_marker)

        if pos >= 0:
            # Find end of the paragraph containing this marker
            end_p = html.find("</p>", pos)
            if end_p >= 0:
                insert_pos = end_p + 4
                html = html[:insert_pos] + "\n" + fig_block + html[insert_pos:]
                print(f"    Inserted {fig_key}: {caption}")
        else:
            print(f"    [SKIP] Marker not found for {fig_key}: {marker[:40]}...")

    return html


def process_references(refs_md):
    """Process references section."""
    if not refs_md:
        return ""
    refs_md = re.sub(r'^##\s+参考文献\s*\n+', '', refs_md)
    lines = refs_md.strip().split('\n')
    refs_html = '<div class="references-section">\n<h2>参考文献</h2>\n'
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', line)
        refs_html += f'<p class="reference-item">{line}</p>\n'
    refs_html += '</div>'
    return refs_html


# =============================================================================
# REPORT BUILDER
# =============================================================================
def build_report():
    """Build the report HTML with all figures embedded."""
    print("\n" + "=" * 60)
    print("REBUILDING CONSULTING REPORT")
    print("=" * 60)

    md_text = REPORT_MD.read_text(encoding="utf-8")
    print(f"  Read {len(md_text)} chars from MD")

    report = parse_report_structure(md_text)
    print(f"  Sections: {len(report['sections'])}")

    # Build sections HTML with figures
    sections_html = ""
    for i, section in enumerate(report["sections"]):
        sections_html += build_report_section(section, i)

    # Build figure gallery
    gallery_html = build_figure_gallery()

    html = REPORT_HTML_TEMPLATE.format(
        title=report["title"],
        subtitle=report["subtitle"],
        abstract=report["abstract"],
        toc_html=build_toc(report["sections"]),
        sections_html=sections_html,
        gallery_html=gallery_html,
    )

    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"  Report HTML saved: {os.path.getsize(REPORT_HTML) / 1024:.1f} KB")

    render_pdf(html, REPORT_PDF, str(REPORT_HTML.parent))


def parse_report_structure(md_text):
    """Parse report markdown."""
    title_match = re.match(r'^#\s+(.+)', md_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "研究报告"

    subtitle_match = re.search(r'^##\s+(——.+)', md_text, re.MULTILINE)
    subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

    abstract_match = re.search(r'##\s*摘要\s*\n\n(.+?)(?=\n---|\n##)', md_text, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    sections = []
    section_pattern = re.compile(r'^##\s+(.+?)$', re.MULTILINE)
    matches = list(section_pattern.finditer(md_text))

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        if heading.startswith('——') or heading == '摘要':
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()
        content = re.sub(r'^---\s*$', '', content, flags=re.MULTILINE).strip()
        sections.append({"heading": heading, "content": content})

    return {
        "title": title,
        "subtitle": subtitle,
        "abstract": abstract,
        "sections": sections,
    }


def build_report_section(section, index):
    """Build a single report section with figure embeddings."""
    heading = section["heading"]
    content_html = md_to_html(section["content"])
    content_html = content_html.replace('<table>', '<table class="report-table">')

    # Style blockquotes
    content_html = re.sub(
        r'<blockquote>\s*<p>\s*<strong>案例聚焦',
        '<blockquote class="case-callout"><p><strong>案例聚焦',
        content_html
    )
    content_html = content_html.replace('<blockquote>', '<blockquote class="callout-box">')

    # Insert figures based on section content
    content_html = insert_report_figures(content_html, heading)

    html = ""
    major_sections = ["一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、"]
    is_major = any(heading.startswith(s) for s in major_sections)

    if is_major:
        sec_num = heading.split("、")[0] if "、" in heading else str(index + 1)
        sec_title = heading.split("、", 1)[1] if "、" in heading else heading
        html += f'''
<div class="section-divider">
    <div class="section-divider-content">
        <div class="section-divider-num">{sec_num}</div>
        <div class="section-divider-title">{sec_title}</div>
    </div>
</div>
'''

    html += f'<div class="section-content">\n'
    html += f'<h2 class="section-heading">{heading}</h2>\n'
    html += content_html
    html += '</div>\n'

    return html


def insert_report_figures(html, heading):
    """Insert relevant figures into report sections."""
    # Determine which figures to insert based on section heading
    figure_plan = []

    if "教育视角" in heading or ("教育" in heading and "实践图景" in heading):
        figure_plan = [
            ("浙江省以432", "fig_edu_province"),
            ("小学阶段以890例", "fig_edu_stage"),
            ("语文学科", "fig_edu_subject"),
            ("助学", "fig_edu_scenario"),
            ("智育", "fig_edu_wuyu"),
            ("10个典型", "fig_edu_cluster"),
        ]
    elif "技术视角" in heading or "技术栈" in heading:
        figure_plan = [
            ("数据分析", "fig_tech_tools"),
            ("技术路径", "fig_tech_sankey"),
            ("软件平台", "fig_tech_model"),
            ("技术要素", "fig_tech_network"),
        ]
    elif "产业视角" in heading or "EdTech" in heading:
        figure_plan = [
            ("CR5", "fig_ind_lorenz"),
            ("碎片化", "fig_ind_ecosystem"),
        ]
    elif "区域分析" in heading or "地理" in heading or "数字教育" in heading:
        figure_plan = [
            ("浙江省", "fig_geo_inequality"),
        ]
    elif "理论视角" in heading or "智慧教育框架" in heading:
        figure_plan = [
            ("三赋能", "fig_fw_istar"),
            ("三重境界", "fig_fw_three_realms"),
            ("数字教学法", "fig_fw_pedagogy"),
            ("深度学习", "fig_fw_innovation"),
            ("Gen4", "fig_fw_bubble"),
        ]
    elif "典型案例" in heading or "深度剖析" in heading:
        figure_plan = [
            ("iSTAR", "fig_fw_istar"),
        ]
    elif "深度洞察" in heading or "真金" in heading or "数据深挖" in heading:
        figure_plan = [
            ("经济水平", "fig_deep_shap"),
            ("Cram", "fig_deep_cramers"),
            ("Spearman", "fig_deep_ca"),
            ("多工具组合", "fig_deep_mca"),
            ("先行者", "fig_deep_anova"),
            ("创新深度评分", "fig_deep_dashboard"),
        ]
    elif "政策建议" in heading or "结论" in heading or "总结" in heading:
        figure_plan = [
            ("技术编排", "fig_deep_dashboard"),
        ]

    for marker, fig_key in figure_plan:
        if fig_key not in REPORT_FIGURES:
            continue
        filename, caption = REPORT_FIGURES[fig_key]
        fig_block = fig_html(filename, caption, for_paper=False)
        if not fig_block:
            continue

        pos = html.find(marker)
        if pos >= 0:
            end_p = html.find("</p>", pos)
            if end_p >= 0:
                insert_pos = end_p + 4
                html = html[:insert_pos] + "\n" + fig_block + html[insert_pos:]
                print(f"    [{heading[:10]}] Inserted {fig_key}")

    return html


def build_toc(sections):
    """Build table of contents."""
    toc = '<div class="toc-page">\n<h2 class="toc-title">目　录</h2>\n<div class="toc-entries">\n'
    for section in sections:
        toc += f'<div class="toc-l1"><span class="toc-text">{section["heading"]}</span></div>\n'
    toc += '</div>\n</div>\n'
    return toc


def build_figure_gallery():
    """Build the appendix figure gallery with ALL high-quality figures."""
    gallery_items = [
        ("fig_a01_province_map.png", "A1. 案例省域分布地图"),
        ("fig_a02_stage_waffle.png", "A2. 学段分布华夫饼图"),
        ("fig_a03_subject_lollipop.png", "A3. 学科渗透棒棒糖图"),
        ("fig_a04_scenario_treemap.png", "A4. 应用场景树图"),
        ("fig_a05_five_edu_radar.png", "A5. 五育融合雷达图"),
        ("fig_b01_top20_tools.png", "B1. AI工具使用频次Top20"),
        ("fig_b02_tech_sankey.png", "B2. 技术路径桑基图"),
        ("fig_b03_model_landscape.png", "B3. AI大模型景观图"),
        ("fig_b04_cooccurrence_network.png", "B4. 技术共现网络"),
        ("fig_c01_umap_clusters.png", "C1. UMAP聚类可视化"),
        ("fig_d01_lorenz_curve.png", "D1. 市场集中度洛伦兹曲线"),
        ("fig_d02_ecosystem_treemap.png", "D2. EdTech产品生态树图"),
        ("fig_e01_sanfuneng_istar.png", "E1. 三赋能与iSTAR框架"),
        ("fig_e02_innovation_ridgeline.png", "E2. 创新深度岭线图"),
        ("fig_e03_three_realms.png", "E3. 智慧教育三重境界"),
        ("fig_e04_digital_pedagogy_radar.png", "E4. 数字教学法雷达图"),
        ("fig_e05_techgen_scenario_bubble.png", "E5. 技术代际-场景气泡图"),
        ("fig_e06_dashboard.png", "E6. 研究综合仪表板"),
        ("fig_f01_rf_shap.png", "F1. 随机森林SHAP分析"),
        ("fig_f02_cramers_v.png", "F2. Cramér's V关联热力图"),
        ("fig_f03_correspondence_biplot.png", "F3. 对应分析双标图"),
        ("fig_f04_mca_biplot.png", "F4. 多重对应分析双标图"),
        ("fig_f05_geographic_inequality.png", "F5. 地理不平等分析"),
        ("fig_f06_cluster_anova.png", "F6. 聚类方差分析"),
    ]

    html = ""
    for filename, caption in gallery_items:
        path = f"{FIG_ABS}/{filename}"
        if os.path.exists(path):
            html += f'''
<div class="gallery-item">
    <img src="file://{path}" alt="{caption}">
    <p class="gallery-caption">{caption}</p>
</div>
'''
    return html


# =============================================================================
# PDF RENDERER
# =============================================================================
def render_pdf(html, pdf_path, base_url):
    """Render HTML to PDF using Chrome headless."""
    print(f"  Rendering PDF: {pdf_path.name}")
    import subprocess
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # Write HTML to temp file for Chrome to read
    html_path = Path(base_url) / "_temp_render.html"
    html_path.write_text(html, encoding="utf-8")
    try:
        result = subprocess.run([
            chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file://{html_path}"
        ], capture_output=True, text=True, timeout=120)
        size = os.path.getsize(pdf_path) if pdf_path.exists() else 0
        print(f"  PDF saved: {size / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"  [ERROR] PDF render failed: {e}")
    finally:
        html_path.unlink(missing_ok=True)


# =============================================================================
# HTML TEMPLATES
# =============================================================================
PAPER_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
/* ============================================
   ACADEMIC PAPER — REDESIGNED v2
   ============================================ */

@page {{
    size: A4;
    margin: 2.5cm 2cm 2cm 2.5cm;
    @top-center {{
        content: "AI赋能基础教育的实践图景与产业生态";
        font-family: 'Songti SC', 'SimSun', serif;
        font-size: 9pt;
        color: #666;
        border-bottom: 0.5pt solid #ccc;
        padding-bottom: 4pt;
    }}
    @bottom-center {{
        content: counter(page);
        font-family: 'Times New Roman', serif;
        font-size: 10pt;
        color: #333;
    }}
}}

@page:first {{
    @top-center {{ content: none; }}
    @bottom-center {{ content: none; }}
}}

body {{
    font-family: 'Songti SC', 'SimSun', serif;
    font-size: 12pt;
    line-height: 1.8;
    color: #1a1a1a;
    text-align: justify;
    orphans: 3;
    widows: 3;
}}

/* TITLE PAGE */
.title-page {{
    page-break-after: always;
    text-align: center;
    padding-top: 5cm;
}}
.paper-title {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 24pt;
    font-weight: 700;
    line-height: 1.4;
    color: #111;
    margin-bottom: 0.3cm;
    letter-spacing: 0.05em;
}}
.paper-subtitle {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 15pt;
    font-weight: 400;
    color: #444;
    margin-bottom: 2.5cm;
}}
.paper-meta {{
    font-family: 'Songti SC', serif;
    font-size: 11pt;
    color: #666;
    margin-top: 3cm;
}}

/* ABSTRACT PAGE */
.abstract-page {{
    page-break-after: always;
}}
.abstract-section {{
    margin-bottom: 1.5cm;
}}
.abstract-heading {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 14pt;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5cm;
    color: #111;
}}
.abstract-text {{
    font-family: 'Songti SC', 'SimSun', serif;
    font-size: 11pt;
    line-height: 1.7;
    text-indent: 2em;
    text-align: justify;
    color: #222;
}}
.abstract-text-en {{
    font-family: 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.6;
    text-indent: 2em;
    text-align: justify;
    color: #222;
}}
.keywords-label {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-weight: 700;
    font-size: 11pt;
}}
.keywords-label-en {{
    font-family: 'Times New Roman', serif;
    font-weight: 700;
    font-size: 10.5pt;
}}
.keywords-text {{
    font-family: 'Songti SC', serif;
    font-size: 11pt;
    margin-bottom: 1cm;
    margin-top: 0.3cm;
}}
.keywords-text-en {{
    font-family: 'Times New Roman', serif;
    font-size: 10.5pt;
    margin-bottom: 1cm;
    margin-top: 0.3cm;
}}
.abstract-divider {{
    border: none;
    border-top: 1pt solid #ddd;
    margin: 1cm 2cm;
}}

/* HEADINGS */
h2 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 17pt;
    font-weight: 700;
    color: #111;
    margin-top: 1.5cm;
    margin-bottom: 0.5cm;
    page-break-after: avoid;
    text-indent: 0;
}}
h3 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 14pt;
    font-weight: 700;
    color: #222;
    margin-top: 1cm;
    margin-bottom: 0.4cm;
    page-break-after: avoid;
    text-indent: 0;
}}
h4 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 12.5pt;
    font-weight: 700;
    color: #333;
    margin-top: 0.8cm;
    margin-bottom: 0.3cm;
    page-break-after: avoid;
    text-indent: 0;
}}

/* BODY TEXT */
p {{
    text-indent: 2em;
    margin-bottom: 0.2cm;
    line-height: 1.8;
}}
strong {{
    font-family: 'Source Han Sans CN', 'Songti SC', serif;
    font-weight: 700;
}}
p em, p i {{
    font-family: 'Times New Roman', serif;
}}

/* THREE-LINE TABLES */
.three-line-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.8cm auto;
    font-size: 10pt;
    line-height: 1.5;
    page-break-inside: avoid;
}}
.three-line-table thead {{
    border-top: 2pt solid #111;
    border-bottom: 1pt solid #111;
}}
.three-line-table thead th {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-weight: 700;
    padding: 6pt 8pt;
    text-align: center;
    font-size: 10pt;
}}
.three-line-table tbody {{
    border-bottom: 2pt solid #111;
}}
.three-line-table tbody td {{
    padding: 5pt 8pt;
    text-align: center;
    font-size: 10pt;
    font-family: 'Songti SC', serif;
}}
.three-line-table tbody tr:nth-child(even) {{
    background-color: #fafafa;
}}
.table-container {{
    margin: 0.8cm 0;
    page-break-inside: avoid;
}}
.table-caption {{
    text-align: center;
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 10.5pt;
    font-weight: 700;
    margin-bottom: 0.3cm;
    text-indent: 0;
}}

/* FIGURES — NEW */
.figure-block {{
    margin: 1cm 0;
    text-align: center;
    page-break-inside: avoid;
}}
.figure-block img {{
    max-width: 100%;
    height: auto;
    max-height: 18cm;
    border: 0.5pt solid #ddd;
}}
.fig-caption {{
    text-align: center;
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 10.5pt;
    font-weight: 700;
    margin-top: 0.3cm;
    color: #111;
    text-indent: 0;
}}
.figure-caption {{
    margin: 0.8cm 0;
    page-break-inside: avoid;
}}
.caption-text {{
    text-align: center;
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 10.5pt;
    font-weight: 700;
    margin-bottom: 0.3cm;
    text-indent: 0;
}}

/* BLOCKQUOTES */
blockquote {{
    margin: 0.5cm 1cm;
    padding: 0.3cm 0.8cm;
    border-left: 3pt solid #ddd;
    background-color: #fafafa;
    font-size: 11pt;
    page-break-inside: avoid;
}}
blockquote p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
}}

/* LISTS */
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

/* REFERENCES */
.references-section {{
    margin-top: 1.5cm;
    page-break-before: always;
}}
.references-section h2 {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 17pt;
    font-weight: 700;
    margin-bottom: 0.8cm;
    text-align: center;
}}
.reference-item {{
    font-size: 10pt;
    line-height: 1.5;
    text-indent: -2em;
    padding-left: 2em;
    margin-bottom: 0.2cm;
    font-family: 'Songti SC', serif;
    text-align: justify;
}}
.reference-item em {{
    font-family: 'Times New Roman', serif;
    font-style: italic;
}}

/* ENDMATTER */
.endmatter {{
    margin-top: 1.5cm;
    font-size: 10.5pt;
    color: #666;
}}
.endmatter p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
    font-size: 10.5pt;
}}

hr {{
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 1cm 0;
}}
code {{
    font-family: 'Menlo', monospace;
    font-size: 9.5pt;
    background-color: #f5f5f5;
    padding: 1pt 3pt;
    border-radius: 2pt;
}}
h2, h3, h4 {{
    page-break-after: avoid;
}}
.table-container, .figure-block, .three-line-table, blockquote {{
    page-break-inside: avoid;
}}
.table-caption p, .caption-text, .abstract-heading, .keywords-text, .keywords-text-en, .fig-caption {{
    text-indent: 0;
}}
</style>
</head>
<body>

<!-- TITLE PAGE -->
<div class="title-page">
    <div class="paper-title">{title}</div>
    <div class="paper-subtitle">{subtitle}</div>
    <div class="paper-meta">2026年</div>
</div>

<!-- ABSTRACT PAGE -->
<div class="abstract-page">
    <div class="abstract-section">
        <div class="abstract-heading">摘　要</div>
        <div class="abstract-text">{cn_abstract}</div>
        <div class="keywords-text">
            <span class="keywords-label">关键词：</span>{cn_keywords}
        </div>
    </div>
    <hr class="abstract-divider">
    <div class="abstract-section">
        <div class="abstract-heading" style="font-family: 'Times New Roman', serif;">Abstract</div>
        <div class="abstract-text-en">{en_abstract}</div>
        <div class="keywords-text-en">
            <span class="keywords-label-en">Keywords: </span>{en_keywords}
        </div>
    </div>
</div>

<!-- BODY -->
<div class="paper-body">
{body_html}
</div>

<!-- REFERENCES -->
{refs_html}

<!-- ENDMATTER -->
<div class="endmatter">
{endmatter_html}
</div>

</body>
</html>'''


REPORT_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
/* ============================================
   CONSULTING REPORT — REDESIGNED v2
   Premium teal/gold design
   ============================================ */

@page {{
    size: A4;
    margin: 2cm 2cm 2.5cm 2cm;
    @top-left {{
        content: "AI赋能基础教育研究报告";
        font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
        font-size: 8pt;
        color: #0D4F4F;
        opacity: 0.7;
    }}
    @top-right {{
        content: "2026";
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 8pt;
        color: #C19A3E;
        opacity: 0.7;
    }}
    @bottom-center {{
        content: counter(page);
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 9pt;
        color: #0D4F4F;
    }}
}}

@page:first {{
    margin: 0;
    @top-left {{ content: none; }}
    @top-right {{ content: none; }}
    @bottom-center {{ content: none; }}
}}

body {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #2a2a2a;
    text-align: justify;
}}

/* COVER */
.cover-page {{
    page-break-after: always;
    width: 210mm;
    height: 297mm;
    position: relative;
    overflow: hidden;
    margin: -2cm;
    padding: 0;
    background: linear-gradient(135deg, #0D4F4F 0%, #0A3D3D 40%, #072E2E 70%, #051F1F 100%);
    color: white;
}}
.cover-overlay {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
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
    color: #C19A3E;
    margin-bottom: 1cm;
}}
.cover-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 30pt;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 0.5cm;
    color: #ffffff;
}}
.cover-subtitle {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 16pt;
    font-weight: 300;
    color: rgba(255,255,255,0.8);
    margin-bottom: 2cm;
}}
.cover-divider {{
    width: 60pt;
    height: 2pt;
    background: #C19A3E;
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
    color: #C19A3E;
    line-height: 1.2;
}}
.cover-stat-label {{
    font-size: 9pt;
    color: rgba(255,255,255,0.65);
}}
.cover-date {{
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 10pt;
    color: rgba(255,255,255,0.5);
    position: absolute;
    bottom: 3cm; left: 3cm;
}}
.cover-accent-line {{
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 4pt;
    background: linear-gradient(90deg, #C19A3E 0%, #D4B76A 50%, transparent 100%);
}}

/* TOC */
.toc-page {{
    page-break-after: always;
    padding-top: 2cm;
}}
.toc-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 22pt;
    font-weight: 700;
    color: #0D4F4F;
    margin-bottom: 1.5cm;
    padding-bottom: 0.3cm;
    border-bottom: 2pt solid #C19A3E;
    text-indent: 0;
}}
.toc-l1 {{
    font-size: 12pt;
    font-weight: 600;
    color: #0D4F4F;
    padding: 0.25cm 0;
    border-bottom: 0.5pt solid #eee;
}}
.toc-text {{
    text-indent: 0;
}}

/* SECTION DIVIDERS */
.section-divider {{
    page-break-before: always;
    page-break-after: avoid;
    width: calc(100% + 4cm);
    margin-left: -2cm;
    margin-right: -2cm;
    margin-top: -2cm;
    padding: 4cm 3cm 3cm 3cm;
    background: linear-gradient(135deg, #0D4F4F 0%, #1A7A7A 100%);
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
    color: #C19A3E;
    line-height: 1;
    margin-bottom: 0.5cm;
}}
.section-divider-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 22pt;
    font-weight: 700;
    color: white;
}}

/* HEADINGS */
h2.section-heading {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #0D4F4F;
    margin-top: 1.2cm;
    margin-bottom: 0.5cm;
    padding-bottom: 0.2cm;
    border-bottom: 1.5pt solid #C19A3E;
    page-break-after: avoid;
    text-indent: 0;
}}
h3 {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 14pt;
    font-weight: 700;
    color: #0D4F4F;
    margin-top: 0.8cm;
    margin-bottom: 0.4cm;
    page-break-after: avoid;
    text-indent: 0;
}}
h4 {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 12pt;
    font-weight: 700;
    color: #1A7A7A;
    margin-top: 0.6cm;
    margin-bottom: 0.3cm;
    page-break-after: avoid;
    text-indent: 0;
}}

/* BODY */
p {{
    margin-bottom: 0.3cm;
    line-height: 1.7;
    text-indent: 2em;
}}
strong {{
    font-weight: 700;
    color: #1a1a1a;
}}

/* TABLES */
.report-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.6cm 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}
.report-table thead {{
    background: #0D4F4F;
    color: white;
}}
.report-table thead th {{
    padding: 8pt 10pt;
    text-align: left;
    font-weight: 600;
    text-indent: 0;
}}
.report-table tbody td {{
    padding: 7pt 10pt;
    border-bottom: 0.5pt solid #e5e5e5;
    text-indent: 0;
}}
.report-table tbody tr:nth-child(even) {{
    background-color: #E8F4F4;
}}

/* CALLOUTS */
.callout-box {{
    margin: 0.6cm 0;
    padding: 0.6cm 0.8cm;
    border-left: 4pt solid #C19A3E;
    background: #FDF6E8;
    border-radius: 0 4pt 4pt 0;
    font-size: 10.5pt;
    page-break-inside: avoid;
}}
.callout-box p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
    color: #333;
}}
.callout-box strong {{
    color: #0D4F4F;
}}
.case-callout {{
    margin: 0.6cm 0;
    padding: 0.6cm 0.8cm;
    border-left: 4pt solid #0D4F4F;
    background: #E8F4F4;
    border-radius: 0 4pt 4pt 0;
    font-size: 10.5pt;
    page-break-inside: avoid;
}}
.case-callout p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
}}
.case-callout strong {{
    color: #0D4F4F;
}}

/* FIGURES */
.figure-embed {{
    margin: 0.8cm 0;
    text-align: center;
    page-break-inside: avoid;
}}
.figure-embed img {{
    max-width: 100%;
    height: auto;
    max-height: 17cm;
    border: 0.5pt solid #e5e5e5;
    border-radius: 3pt;
}}
.figure-caption-text {{
    font-size: 9.5pt;
    color: #0D4F4F;
    font-weight: 600;
    text-align: center;
    margin-top: 0.3cm;
    text-indent: 0;
}}

/* GALLERY */
.figures-page {{
    page-break-before: always;
}}
.figures-page-title {{
    font-family: 'Source Han Sans CN', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #0D4F4F;
    margin-bottom: 0.8cm;
    padding-bottom: 0.2cm;
    border-bottom: 1.5pt solid #C19A3E;
    text-indent: 0;
}}
.gallery-item {{
    margin-bottom: 0.6cm;
    page-break-inside: avoid;
    text-align: center;
}}
.gallery-item img {{
    max-width: 100%;
    height: auto;
    max-height: 15cm;
    border: 0.5pt solid #e5e5e5;
    border-radius: 3pt;
}}
.gallery-caption {{
    font-size: 9.5pt;
    color: #0D4F4F;
    font-weight: 600;
    text-align: center;
    margin-top: 0.2cm;
    margin-bottom: 0.3cm;
    text-indent: 0;
}}

/* ABSTRACT BOX */
.abstract-box {{
    margin: 1cm 0;
    padding: 0.8cm 1cm;
    background: linear-gradient(135deg, #E8F4F4 0%, #f5fafa 100%);
    border: 1pt solid #c5dede;
    border-radius: 4pt;
    font-size: 10.5pt;
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
    color: #0D4F4F;
    margin-bottom: 0.4cm;
    text-indent: 0;
}}

/* LISTS */
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

hr {{
    border: none;
    border-top: 1pt solid #C19A3E;
    margin: 1cm 0;
    opacity: 0.4;
}}

.section-content {{
    margin-bottom: 1cm;
}}

/* ENDMATTER */
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

h2, h3, h4 {{
    page-break-after: avoid;
}}
.report-table, .callout-box, .case-callout, .figure-embed, .gallery-item {{
    page-break-inside: avoid;
}}
</style>
</head>
<body>

<!-- COVER -->
<div class="cover-page">
    <div class="cover-overlay"></div>
    <div class="cover-content">
        <div class="cover-tag">RESEARCH REPORT &middot; 研究报告</div>
        <h1 class="cover-title">{title}</h1>
        <div class="cover-subtitle">{subtitle}</div>
        <div class="cover-divider"></div>
        <div class="cover-stats">
            <div class="cover-stat">
                <div class="cover-stat-value">1,690</div>
                <div class="cover-stat-label">有效案例</div>
            </div>
            <div class="cover-stat">
                <div class="cover-stat-value">1,830</div>
                <div class="cover-stat-label">独立工具/产品</div>
            </div>
            <div class="cover-stat">
                <div class="cover-stat-value">1,726</div>
                <div class="cover-stat-label">企业/机构主体</div>
            </div>
            <div class="cover-stat">
                <div class="cover-stat-value">30</div>
                <div class="cover-stat-label">省级行政区覆盖</div>
            </div>
        </div>
        <div class="cover-date">2026年2月</div>
    </div>
    <div class="cover-accent-line"></div>
</div>

<!-- TOC -->
{toc_html}

<!-- ABSTRACT -->
<div class="abstract-box">
    <div class="abstract-box-title">摘　要</div>
    <p>{abstract}</p>
</div>

<!-- SECTIONS -->
{sections_html}

<!-- FIGURE GALLERY -->
<div class="figures-page">
    <h2 class="figures-page-title">附录：核心数据可视化图表</h2>
    {gallery_html}
</div>

<!-- ENDMATTER -->
<div class="endmatter">
    <p>本报告基于1690个AI教育应用案例的数据分析生成。报告中的分析和建议仅代表研究团队基于数据的学术判断。</p>
    <p>数据截止至案例征集活动结束日期。 &copy; 2026</p>
</div>

</body>
</html>'''


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("COMPREHENSIVE PAPER & REPORT REBUILD")
    print("with all high-quality figures embedded")
    print("=" * 60)

    # Check figures exist
    fig_count = len(list(FIGURES_DIR.glob("fig_*.png")))
    print(f"\nFigures directory: {FIGURES_DIR}")
    print(f"High-quality figures found: {fig_count}")

    # Build paper
    build_paper()

    # Build report
    build_report()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\nPaper HTML: {PAPER_HTML}")
    print(f"Paper PDF:  {PAPER_PDF}")
    print(f"Report HTML: {REPORT_HTML}")
    print(f"Report PDF:  {REPORT_PDF}")


if __name__ == "__main__":
    main()
