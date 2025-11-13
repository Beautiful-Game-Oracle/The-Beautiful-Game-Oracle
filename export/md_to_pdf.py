from fpdf import FPDF
import textwrap

INPUT_MD = "export/elo_explanation.md"
OUTPUT_PDF = "export/elo_explanation.pdf"

def render_markdown_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # use built-in fonts to avoid external font files
    # helpers
    def write_paragraph(text, font_size=11, bold=False):
        font_name = "Arial"
        style = "B" if bold else ""
        pdf.set_font(font_name, style, font_size)
        wrapped = textwrap.wrap(text, width=95)
        for line in wrapped:
            pdf.multi_cell(0, 6, line)
        pdf.ln(2)

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("# "):
            pdf.ln(2)
            write_paragraph(line[2:].strip(), font_size=16, bold=True)
            pdf.ln(1)
        elif line.startswith("## "):
            write_paragraph(line[3:].strip(), font_size=14, bold=True)
        elif line.startswith("### "):
            write_paragraph(line[4:].strip(), font_size=13, bold=True)
        elif line.strip() == "":
            pdf.ln(2)
        else:
            # plain paragraph
            write_paragraph(line, font_size=11)

    pdf.output(pdf_path)

if __name__ == "__main__":
    try:
        render_markdown_to_pdf(INPUT_MD, OUTPUT_PDF)
        print(f"Wrote {OUTPUT_PDF}")
    except Exception as e:
        print("Error creating PDF:", e)
