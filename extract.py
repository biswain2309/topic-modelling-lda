from PyPDF2 import PdfFileReader
import os
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



list_obj = []
page_list = []
text_str = ""

def extract_information(pdf_path):
    
    combined_text = ""

    for filename in os.listdir(pdf_path):

        if filename.endswith('.pdf'):

            pdf_document = f'{pdf_path}{filename}'

        with open(pdf_document, "rb") as filehandle:
            pdf = PdfFileReader(filehandle)
            pages = pdf.getNumPages()

            for i in range(0, pages):
                pageObj = pdf.getPage(i)
                text_str = pageObj.extractText()
                combined_text = combined_text + " " + text_str
            page_list.append(combined_text)


#extraction of data from pdf
extract_information('Covid_medicine/')

# opening the csv file in 'w+' mode 
file = open('f_outputb.csv', 'w+') 
  
# writing the data into the file 
with file:     
    df = pd.DataFrame(page_list, columns=['Article'])
    df.to_csv(file,index=True)

npr = pd.read_csv('f_outputb.csv', error_bad_lines=False)


