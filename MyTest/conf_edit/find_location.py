import fitz  # PyMuPDF


def extract_header_margins_in_cm(pdf_path, page_number=0, header_search_area_ratio=0.1):
    document = fitz.open(pdf_path)
    page = document.load_page(page_number)
    page_height = page.rect.height

    text_blocks = page.get_text("blocks")

    header_left_margin = None
    header_top_margin = None

    for block in text_blocks:
        if len(block) >= 5:
            x0, y0, _, _, _ = block[:5]
            if y0 < page_height * header_search_area_ratio:
                if header_left_margin is None or x0 < header_left_margin:
                    header_left_margin = x0
                if header_top_margin is None or y0 < header_top_margin:
                    header_top_margin = y0

    # 将左边距和上边距从点转换为厘米
    pts_to_cm = 0.03528
    header_left_margin_cm = header_left_margin * pts_to_cm if header_left_margin is not None else None
    header_top_margin_cm = header_top_margin * pts_to_cm if header_top_margin is not None else None

    return header_left_margin_cm, header_top_margin_cm


# 使用示例
pdf_path = "pdf/1.pdf"
left_margin_cm, top_margin_cm = extract_header_margins_in_cm(pdf_path)

if left_margin_cm is not None and top_margin_cm is not None:
    print(f"Header left margin: {left_margin_cm:.2f} cm")
    print(f"Header top margin: {top_margin_cm:.2f} cm")
else:
    print("No header found.")
