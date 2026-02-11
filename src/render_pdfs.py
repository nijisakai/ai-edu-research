#!/usr/bin/env python3
"""Render HTML files to PDF using WeasyPrint."""

import os
import sys
from pathlib import Path

BASE = Path("/Users/sakai/Desktop/产业调研/ai-edu-research")

PAPER_HTML = BASE / "output" / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.html"
PAPER_PDF = BASE / "output" / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.pdf"

REPORT_HTML = BASE / "output" / "report" / "研究报告_AI赋能基础教育实践图景与产业生态分析.html"
REPORT_PDF = BASE / "output" / "report" / "研究报告_AI赋能基础教育实践图景与产业生态分析.pdf"


def render(html_path, pdf_path):
    from weasyprint import HTML
    from weasyprint.text.fonts import FontConfiguration

    print(f"  Reading: {html_path.name}")
    html_text = html_path.read_text(encoding="utf-8")

    font_config = FontConfiguration()
    doc = HTML(string=html_text, base_url=str(html_path.parent))
    print(f"  Rendering PDF...")
    doc.write_pdf(str(pdf_path), font_config=font_config)

    size = os.path.getsize(pdf_path)
    print(f"  PDF saved: {pdf_path.name} ({size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("PDF RENDERING")
    print("=" * 60)

    print("\n[1/2] Paper PDF")
    try:
        render(PAPER_HTML, PAPER_PDF)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[2/2] Report PDF")
    try:
        render(REPORT_HTML, REPORT_PDF)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nDone!")
