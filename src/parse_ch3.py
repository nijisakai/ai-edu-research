import docx
import os

doc = docx.Document('new_reviews/test.docx')

with open('output/ch3_tables.txt', 'w', encoding='utf-8') as f:
    in_target_section = False
    for child in doc.element.body:
        if child.tag.endswith('p'):
            p = docx.text.paragraph.Paragraph(child, doc)
            if '第三章' in p.text and '应用现状' in p.text:
                in_target_section = True
                f.write(f"--- START SECT --- {p.text}\n")
            elif '3.1' in p.text and in_target_section:
                in_target_section = False
                f.write(f"--- END SECT --- {p.text}\n")
        elif child.tag.endswith('tbl') and in_target_section:
            tbl = docx.table.Table(child, doc)
            f.write("TABLE:\n")
            for row in tbl.rows:
                row_text = "|".join([c.text.strip().replace('\n', ' ') for c in row.cells])
                f.write(f"  {row_text}\n")
            f.write("\n")
