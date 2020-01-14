import PyPDF2

input_file = r"D:\Desktop\PBE\Images\31.signed.pdf"
watermark_file = r"D:\Desktop\PBE\Images\Logo1.pdf"

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
        for page in pages:
            page.mergePage(wtpage)
            n_pages.append(page)
            writer = PyPDF2.PdfFileWriter()
        for page in n_pages:
            writer.addPage(page)
        with open('output.pdf', 'wb') as o:
            writer.write(o)
        o.close()
    wt.close()
f.close()
