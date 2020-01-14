# import win32com.client
# from pywintypes import com_error


# # Path to original excel file
# WB_PATH = r'C:\Users\16521\Music\work\file3.xlsx'
# # PDF path when saving
# PATH_TO_PDF = r'C:\Users\16521\Music\work\result.pdf'


# excel = win32com.client.Dispatch("Excel.Application")

# excel.Visible = False

# try:
#     print('Start conversion to PDF')

#     # Open
#     wb = excel.Workbooks.Open(WB_PATH)

#     # Specify the sheet you want to save by index. 1 is the first (leftmost) sheet.
#     ws_index_list = [1]
#     wb.WorkSheets(ws_index_list).Select()

#     # Save
#     wb.ActiveSheet.ExportAsFixedFormat(0, PATH_TO_PDF)
# except com_error as e:
#     print('failed.')
# else:
#     print('Succeeded.')
# finally:
#     wb.Close()
#     excel.Quit()

# from docx2pdf import convert


# convert(r"C:\Users\16521\Music\work\pcfg.docx", r"C:\Users\16521\Music\work\pcfg.pdf")
import comtypes.client


def PPTtoPDF(inputFileName, outputFileName, formatType = 32):
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1

    if outputFileName[-3:] != 'pdf':
        outputFileName = outputFileName + ".pdf"
    deck = powerpoint.Presentations.Open(inputFileName)
    deck.SaveAs(outputFileName, formatType) # formatType = 32 for ppt to pdf
    deck.Close()
    powerpoint.Quit()

PPTtoPDF(r"C:\Users\16521\Music\work\Chuong4.ppt",r"C:\Users\16521\Music\work\Chuong4.pdf")