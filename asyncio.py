import docx
import zipfile
import re
import base64

filename = "bt.docx"
txtname = "result.txt"
# with open(filename, 'rb') as f:
#     content = f.read()
#     string = base64.b64encode(content)
# f.close()

# data = base64.b64decode(string)
# print(type(data))

# docx = zipfile.ZipFile(data)
# content = docx.read('word/document.xml').decode('utf-8')
# cleaned = re.sub('<(.|\n)*?>','',content)
# print(type(cleaned))

# with open(txtname, 'r', encoding='utf-8') as f:
#     string = f.read()

# print(string)
# f.close()
typename = 'docx'
if typename == 'pdf':
    print(1)
elif typename == 'docx':
    print(2)
elif typename == 'txt':
    print(3)
else:
    print(4)