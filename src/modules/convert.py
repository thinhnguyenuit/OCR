import PyPDF2
from reportlab.pdfgen import canvas
from PIL import Image

input_file = r"D:\Desktop\PBE\Images\31.signed.pdf"
watermark_file = r"D:\Desktop\PBE\Images\logo2.pdf"
img_path = r"D:\Desktop\PBE\Images\logo2.png"

img = Image.open(img_path)
img = img.convert("RGBA")

c = canvas.Canvas(watermark_file)
c.drawImage(img_path, -35, 20, 750, 750, mask='auto', preserveAspectRatio=True)
c.save()
with open(input_file, 'rb') as f:
    pdf = PyPDF2.PdfFileReader(f)
    with open(watermark_file, 'rb') as wt:
        watermark = PyPDF2.PdfFileReader(wt)
        numpage = pdf.getNumPages()
        pages = []
        for i in range(0, numpage):
            page = pdf.getPage(i)
            pages.append(page)
        wtpage = watermark.getPage(0)
        n_pages = []
        # pages.reverse()
        for page in pages:
            page.mergePage(wtpage)
            n_pages.append(page)
            writer = PyPDF2.PdfFileWriter()
        n_pages.reverse()
        for page in n_pages:
            writer.addPage(page)
    wt.close()
f.close()

with open(input_file, 'wb') as o:
    writer.write(o)
o.close()