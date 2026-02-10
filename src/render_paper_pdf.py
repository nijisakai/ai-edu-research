#!/usr/bin/env python3
"""
Render academic paper Markdown to beautifully typeset PDF.
Uses weasyprint for HTML->PDF conversion with academic paper styling.
"""

import re
import os
import sys
import markdown
from pathlib import Path

# Paths
PAPER_MD = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/paper/论文_AI赋能基础教育的实践图景与产业生态.md")
PAPER_HTML = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/paper/论文_AI赋能基础教育的实践图景与产业生态.html")
PAPER_PDF = Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output/paper/论文_AI赋能基础教育的实践图景与产业生态.pdf")


def read_markdown():
    """Read the markdown paper file."""
    with open(PAPER_MD, "r", encoding="utf-8") as f:
        return f.read()


def parse_paper_structure(md_text):
    """Parse the markdown into structured sections for custom rendering."""
    # Extract title (first H1)
    title_match = re.match(r'^#\s+(.+)', md_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "论文"

    # Extract subtitle (## starting with ——)
    subtitle_match = re.search(r'^##\s+(——.+)', md_text, re.MULTILINE)
    subtitle = subtitle_match.group(1).strip() if subtitle_match else ""

    # Extract Chinese abstract
    cn_abstract_match = re.search(
        r'\*\*摘要\*\*[：:]\s*(.+?)(?=\n\n\*\*关键词\*\*)',
        md_text, re.DOTALL
    )
    cn_abstract = cn_abstract_match.group(1).strip() if cn_abstract_match else ""

    # Extract Chinese keywords
    cn_keywords_match = re.search(
        r'\*\*关键词\*\*[：:]\s*(.+?)(?=\n\n)',
        md_text, re.DOTALL
    )
    cn_keywords = cn_keywords_match.group(1).strip() if cn_keywords_match else ""

    # Extract English abstract
    en_abstract_match = re.search(
        r'\*\*Abstract\*\*[：:]\s*(.+?)(?=\n\n\*\*Keywords\*\*)',
        md_text, re.DOTALL
    )
    en_abstract = en_abstract_match.group(1).strip() if en_abstract_match else ""

    # Extract English keywords
    en_keywords_match = re.search(
        r'\*\*Keywords\*\*[：:]\s*(.+?)(?=\n\n)',
        md_text, re.DOTALL
    )
    en_keywords = en_keywords_match.group(1).strip() if en_keywords_match else ""

    # Extract body (everything after keywords until references)
    # Find the start of body content (after the English keywords)
    body_start = md_text.find("## 一、引言")
    refs_start = md_text.find("## 参考文献")

    body_md = ""
    refs_md = ""
    endmatter_md = ""

    if body_start > 0 and refs_start > 0:
        body_md = md_text[body_start:refs_start].strip()
        after_refs = md_text[refs_start:]
        # Split references from end matter
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


def md_to_html_body(md_text):
    """Convert markdown text to HTML using python-markdown."""
    extensions = ['tables', 'footnotes', 'toc', 'smarty']
    return markdown.markdown(md_text, extensions=extensions)


def process_body_html(html):
    """Post-process body HTML for academic paper styling."""
    # Convert blockquote-based tables/figures to proper elements
    # Figures in blockquotes: > **图X ...**
    # Tables in blockquotes with actual table content

    # Add class to tables
    html = html.replace('<table>', '<table class="three-line-table">')

    # Style figure references [图X]
    html = re.sub(r'\[图(\d+)\]', r'<span class="fig-ref">[图\1]</span>', html)

    # Convert > **图X ...** to figure captions
    html = re.sub(
        r'<blockquote>\s*<p>\s*<strong>(图\d+[^<]*)</strong>\s*</p>',
        r'<div class="figure-caption"><p class="caption-text"><strong>\1</strong></p>',
        html
    )

    # Handle blockquote content that represents figures/tables
    # Make blockquote tables look better
    html = re.sub(
        r'<blockquote>\s*<p>\s*<strong>(表\d+[^<]*)</strong>\s*</p>',
        r'<div class="table-container"><p class="table-caption"><strong>\1</strong></p>',
        html
    )

    # Close any opened figure/table containers at blockquote end
    # This is a simplified approach
    html = html.replace('</blockquote>', '</div>')

    return html


def process_references_html(refs_md):
    """Process references section for hanging indent styling."""
    if not refs_md:
        return ""
    # Remove the heading
    refs_md = re.sub(r'^##\s+参考文献\s*\n+', '', refs_md)
    # Convert each reference line
    lines = refs_md.strip().split('\n')
    refs_html = '<div class="references-section">\n<h2>参考文献</h2>\n'
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Process markdown formatting within references
        line = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', line)
        refs_html += f'<p class="reference-item">{line}</p>\n'
    refs_html += '</div>'
    return refs_html


def build_full_html(paper):
    """Build the complete HTML document."""
    # Convert body markdown to HTML
    body_html = md_to_html_body(paper["body_md"])
    body_html = process_body_html(body_html)

    # Process references
    refs_html = process_references_html(paper["refs_md"])

    # Process endmatter
    endmatter_html = ""
    if paper["endmatter_md"]:
        endmatter_html = md_to_html_body(paper["endmatter_md"])

    short_title = "AI赋能基础教育的实践图景与产业生态"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{paper['title']}</title>
<style>
/* ============================================
   ACADEMIC PAPER STYLESHEET
   A4 format, professional typesetting
   ============================================ */

/* Page setup */
@page {{
    size: A4;
    margin: 2.5cm 2cm 2cm 2.5cm;

    @top-center {{
        content: "{short_title}";
        font-family: 'Songti SC', 'SimSun', serif;
        font-size: 9pt;
        color: #666;
        border-bottom: 0.5pt solid #ccc;
        padding-bottom: 4pt;
        margin-bottom: 8pt;
    }}

    @bottom-center {{
        content: counter(page);
        font-family: 'Times New Roman', serif;
        font-size: 10pt;
        color: #333;
    }}
}}

@page:first {{
    @top-center {{
        content: none;
    }}
    @bottom-center {{
        content: none;
    }}
}}

/* Base typography */
body {{
    font-family: 'Songti SC', 'SimSun', serif;
    font-size: 12pt;
    line-height: 1.8;
    color: #1a1a1a;
    text-align: justify;
    -webkit-hyphens: auto;
    hyphens: auto;
    orphans: 3;
    widows: 3;
}}

/* ---- TITLE PAGE ---- */
.title-page {{
    page-break-after: always;
    text-align: center;
    padding-top: 6cm;
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
    letter-spacing: 0.02em;
}}

.paper-authors {{
    font-family: 'Songti SC', 'SimSun', serif;
    font-size: 13pt;
    color: #333;
    margin-bottom: 0.5cm;
}}

.paper-affiliation {{
    font-family: 'Songti SC', 'SimSun', serif;
    font-size: 11pt;
    color: #666;
    margin-bottom: 3cm;
}}

.paper-date {{
    font-family: 'Songti SC', serif;
    font-size: 11pt;
    color: #666;
}}

/* ---- ABSTRACT PAGE ---- */
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
    margin-bottom: 0.4cm;
    color: #222;
}}

.abstract-text-en {{
    font-family: 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.6;
    text-indent: 2em;
    text-align: justify;
    margin-bottom: 0.4cm;
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
    font-family: 'Songti SC', 'SimSun', serif;
    font-size: 11pt;
    line-height: 1.6;
    margin-bottom: 1cm;
}}

.keywords-text-en {{
    font-family: 'Times New Roman', serif;
    font-size: 10.5pt;
    line-height: 1.6;
    margin-bottom: 1cm;
}}

.abstract-divider {{
    border: none;
    border-top: 1pt solid #ddd;
    margin: 1cm 2cm;
}}

/* ---- HEADINGS ---- */
h2 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #111;
    margin-top: 1.5cm;
    margin-bottom: 0.5cm;
    line-height: 1.4;
    page-break-after: avoid;
}}

h3 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 15pt;
    font-weight: 700;
    color: #222;
    margin-top: 1cm;
    margin-bottom: 0.4cm;
    line-height: 1.4;
    page-break-after: avoid;
}}

h4 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 13pt;
    font-weight: 700;
    color: #333;
    margin-top: 0.8cm;
    margin-bottom: 0.3cm;
    line-height: 1.4;
    page-break-after: avoid;
}}

/* ---- BODY TEXT ---- */
p {{
    text-indent: 2em;
    margin-bottom: 0.2cm;
    line-height: 1.8;
}}

/* English text within body */
p em, p i {{
    font-family: 'Times New Roman', serif;
}}

/* Bold text */
strong {{
    font-family: 'Source Han Sans CN', 'Songti SC', serif;
    font-weight: 700;
}}

/* ---- THREE-LINE TABLES ---- */
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
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-weight: 700;
    padding: 6pt 8pt;
    text-align: center;
    font-size: 10pt;
    color: #111;
}}

.three-line-table tbody {{
    border-bottom: 2pt solid #111;
}}

.three-line-table tbody td {{
    padding: 5pt 8pt;
    text-align: center;
    font-size: 10pt;
    font-family: 'Songti SC', 'SimSun', serif;
    color: #222;
}}

.three-line-table tbody tr:nth-child(even) {{
    background-color: #fafafa;
}}

/* ---- TABLE & FIGURE CAPTIONS ---- */
.table-container {{
    margin: 0.8cm 0;
    page-break-inside: avoid;
}}

.table-caption {{
    text-align: center;
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 10.5pt;
    font-weight: 700;
    margin-bottom: 0.3cm;
    color: #111;
    text-indent: 0;
}}

.figure-caption {{
    margin: 0.8cm 0;
    page-break-inside: avoid;
}}

.caption-text {{
    text-align: center;
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 10.5pt;
    font-weight: 700;
    margin-bottom: 0.3cm;
    color: #111;
    text-indent: 0;
}}

/* Figure references */
.fig-ref {{
    font-family: 'Songti SC', serif;
    color: #333;
}}

/* ---- BLOCKQUOTES (for quoted text, models, etc.) ---- */
blockquote {{
    margin: 0.5cm 1cm;
    padding: 0.3cm 0.8cm;
    border-left: 3pt solid #ddd;
    background-color: #fafafa;
    font-size: 11pt;
    line-height: 1.6;
    page-break-inside: avoid;
}}

blockquote p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
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

/* ---- REFERENCES ---- */
.references-section {{
    margin-top: 1.5cm;
    page-break-before: always;
}}

.references-section h2 {{
    font-family: 'Source Han Sans CN', 'PingFang SC', sans-serif;
    font-size: 18pt;
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
    font-family: 'Songti SC', 'SimSun', serif;
    text-align: justify;
}}

.reference-item em {{
    font-family: 'Times New Roman', serif;
    font-style: italic;
}}

/* ---- ENDMATTER ---- */
.endmatter {{
    margin-top: 1.5cm;
    font-size: 10.5pt;
    color: #666;
    text-align: left;
}}

.endmatter p {{
    text-indent: 0;
    margin-bottom: 0.2cm;
    font-size: 10.5pt;
}}

/* ---- HORIZONTAL RULES ---- */
hr {{
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 1cm 0;
}}

/* ---- CODE (if any) ---- */
code {{
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 9.5pt;
    background-color: #f5f5f5;
    padding: 1pt 3pt;
    border-radius: 2pt;
}}

/* Avoid page break issues */
h2, h3, h4 {{
    page-break-after: avoid;
}}

.table-container, .figure-caption, .three-line-table, blockquote {{
    page-break-inside: avoid;
}}

/* Override text-indent for special elements */
.table-caption p, .caption-text, .abstract-heading,
.keywords-text, .keywords-text-en {{
    text-indent: 0;
}}

</style>
</head>
<body>

<!-- ======== TITLE PAGE ======== -->
<div class="title-page">
    <div class="paper-title">{paper['title']}</div>
    <div class="paper-subtitle">{paper['subtitle']}</div>
    <div class="paper-authors">&nbsp;</div>
    <div class="paper-affiliation">&nbsp;</div>
    <div class="paper-date">2026年</div>
</div>

<!-- ======== ABSTRACT PAGE ======== -->
<div class="abstract-page">
    <!-- Chinese Abstract -->
    <div class="abstract-section">
        <div class="abstract-heading">摘　要</div>
        <div class="abstract-text">{paper['cn_abstract']}</div>
        <div class="keywords-text">
            <span class="keywords-label">关键词：</span>{paper['cn_keywords']}
        </div>
    </div>

    <hr class="abstract-divider">

    <!-- English Abstract -->
    <div class="abstract-section">
        <div class="abstract-heading" style="font-family: 'Times New Roman', serif;">Abstract</div>
        <div class="abstract-text-en">{paper['en_abstract']}</div>
        <div class="keywords-text-en">
            <span class="keywords-label-en">Keywords: </span>{paper['en_keywords']}
        </div>
    </div>
</div>

<!-- ======== BODY ======== -->
<div class="paper-body">
{body_html}
</div>

<!-- ======== REFERENCES ======== -->
{refs_html}

<!-- ======== ENDMATTER ======== -->
<div class="endmatter">
{endmatter_html}
</div>

</body>
</html>
"""
    return html


def main():
    print("=" * 60)
    print("Academic Paper PDF Renderer")
    print("=" * 60)

    # Read markdown
    print(f"\n[1/4] Reading markdown: {PAPER_MD}")
    md_text = read_markdown()
    print(f"      Read {len(md_text)} characters")

    # Parse structure
    print("[2/4] Parsing paper structure...")
    paper = parse_paper_structure(md_text)
    print(f"      Title: {paper['title']}")
    print(f"      Subtitle: {paper['subtitle']}")
    print(f"      CN Abstract: {len(paper['cn_abstract'])} chars")
    print(f"      EN Abstract: {len(paper['en_abstract'])} chars")
    print(f"      Body: {len(paper['body_md'])} chars")
    print(f"      References: {len(paper['refs_md'])} chars")

    # Build HTML
    print("[3/4] Building HTML...")
    html = build_full_html(paper)

    # Save HTML
    print(f"      Saving HTML: {PAPER_HTML}")
    with open(PAPER_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"      HTML size: {os.path.getsize(PAPER_HTML) / 1024:.1f} KB")

    # Convert to PDF
    print(f"[4/4] Converting to PDF: {PAPER_PDF}")
    try:
        import weasyprint
        from weasyprint import HTML
        from weasyprint.text.fonts import FontConfiguration

        font_config = FontConfiguration()
        doc = HTML(string=html, base_url=str(PAPER_HTML.parent))
        doc.write_pdf(
            str(PAPER_PDF),
            font_config=font_config,
        )
        pdf_size = os.path.getsize(PAPER_PDF)
        print(f"      PDF size: {pdf_size / 1024:.1f} KB ({pdf_size / (1024*1024):.2f} MB)")
        print("\n[DONE] Paper PDF generated successfully!")
    except Exception as e:
        print(f"\n[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
