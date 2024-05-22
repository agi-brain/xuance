from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# 将SVG文件转换为ReportLab图形对象
drawing = svg2rlg("Train_Episode_Score.svg")

# 将图形对象渲染到PDF文件
renderPDF.drawToFile(drawing, "train-Episode-Rewards.pdf")