#!/usr/bin/env python3
"""Generate premium academic paper HTML."""
import pathlib, textwrap

BASE = pathlib.Path("/Users/sakai/Desktop/产业调研/ai-edu-research/output")
MD = (BASE / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.md").read_text("utf-8")

# We'll build the HTML by converting the markdown content into proper academic HTML
# with all the CSS styling inline

out_path = BASE / "paper" / "论文_AI赋能基础教育的实践图景与产业生态.html"

# Read the full markdown to extract content
lines = MD.split('\n')

print(f"Read {len(lines)} lines from markdown")
print("Building paper HTML...")

# Write the HTML
out_path.write_text(open("/dev/stdin").read() if False else "", "utf-8")
# We'll use a different approach - write directly
print("Done setup")
