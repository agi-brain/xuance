import os
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.lib.units import cm,mm

def create_watermark( width, height, position):
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=(width, height))
    # 根据图片设置页边距
    top_margin = 1.66 * cm  # 上边距2厘米
    bottom_margin = 1.27 * cm  # 下边距1.27厘米
    left_margin = 1.7 * cm  # 左边距1.7厘米
    right_margin = 2.54 * cm  # 右边距2.54厘米
    # 设置字体和字号
    c.setFont("Times-Roman", 10)

    if position == 'header':
        # 页眉的文本和格式设置
        # c.drawString(40, height - 40,
        #              "Proceedings of the 3rd Conference on Fully Actuated System Theory and Applications")  # 主标题
        # c.drawString(40, height - 55, "May 10-12, 2024, Shenzhen, China")  # 日期和地点
        c.drawString(left_margin, height - top_margin+14 ,
                     "Proceedings of the 3rd Conference on Fully Actuated System Theory and Applications")  # 主标题
        c.drawString(left_margin, height - top_margin+2.2,
                     "May 10-12, 2024, Shenzhen, China")  # 日期和地点
    elif position == 'footer':
        # 页脚的文本和格式设置
        c.drawString(60, height - 80,"")  # 页脚文本位置可以根据需要调整

    c.save()
    packet.seek(0)

    # 返回PdfReader对象
    from PyPDF2 import PdfReader
    return PdfReader(packet)

def add_header_footer(input_pdf, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        width = float(page.mediabox[2])
        height = float(page.mediabox[3])
        if i==0:

            header_watermark = create_watermark( width, height, 'header')
            footer_watermark = create_watermark(width, height, 'footer')

            page.merge_page(header_watermark.pages[0])
            page.merge_page(footer_watermark.pages[0])
        writer.add_page(page)


    with open(output_pdf, "wb") as output:
        writer.write(output)

def process_pdfs(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.pdf'):
                input_path = os.path.join(foldername, filename)
                output_path = os.path.join(foldername, f"{filename}")
                add_header_footer(input_path, output_path)
                print(f"{input_path} finished!")

# 使用示例
root_folder = './pdf/'

process_pdfs(root_folder)