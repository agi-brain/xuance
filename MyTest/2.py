import os
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def create_watermark(text, width, height):
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))
    can.drawString(30, height - 40, text)  # 页眉
    can.drawString(30, 40, text)  # 页脚
    can.save()
    packet.seek(0)
    return PdfReader(packet)

def add_header_footer(input_pdf, output_pdf, header_text, footer_text):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        width = page.mediabox[2]
        height = page.mediabox[3]

        watermark = create_watermark(f"Header: {header_text}", width, height)
        footer = create_watermark(f"Footer: {footer_text}", width, height)

        page.merge_page(watermark.pages[0])
        page.merge_page(footer.pages[0])
        writer.add_page(page)

    with open(output_pdf, "wb") as output:
        writer.write(output)

def process_pdfs(root_folder, header_text, footer_text):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.pdf'):
                input_path = os.path.join(foldername, filename)
                output_path = os.path.join(foldername, f"modified_{filename}")
                add_header_footer(input_path, output_path, header_text, footer_text)
                print(f"Processed {input_path}")

# 使用示例
root_folder = '/path/to/root/folder'
header_text = 'This is the header'
footer_text = 'This is the footer'
process_pdfs(root_folder, header_text, footer_text)
