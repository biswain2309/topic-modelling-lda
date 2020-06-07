from PyPDF2 import PdfFileReader
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF



list_obj = []
page_list = []
text_str = ""
file_size_list = []
name_of_file = []

def extract_information(pdf_path):
    
    combined_text = ""

    for filename in os.listdir(pdf_path):

        if filename.endswith('.pdf'):

            pdf_document = f'{pdf_path}{filename}'
            name_of_file.append(filename)
            df_name_of_file = pd.DataFrame(name_of_file, columns=['PDF Name'])

        with open(pdf_document, "rb") as filehandle:
            pdf = PdfFileReader(filehandle)
            pages = pdf.getNumPages()

            for i in range(0, pages):
                pageObj = pdf.getPage(i)
                text_str = pageObj.extractText()
                combined_text = combined_text + " " + text_str
            page_list.append(combined_text)
    

    # opening the csv file in 'w+' mode 
    file = open('f_outputc.csv', 'w+') 
    
    # writing the data into the file 
    with file:     
        df = pd.DataFrame(page_list, columns=['Article'])
        df.to_csv(file,index=False)

    npr = pd.read_csv('f_outputc.csv', error_bad_lines=False)

    tfidf = TfidfVectorizer(max_df=0.8,min_df=5,stop_words='english')

    dtm = tfidf.fit_transform(npr['Article'])

    nmf_model = NMF(n_components=7,random_state=50)
    nmf_model.fit(dtm)

    #returns index positions that sort the array
    #checking which word in the topic has high probability
    for i,topic in enumerate(nmf_model.components_):
        print(f"THE TOP 15 WORDS FOR TOPIC #{i}")
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
        print('\n')

    #probability of a document belonging to a topic
    topic_results = nmf_model.transform(dtm)


    npr['Topic'] = topic_results.argmax(axis=1)

    topic_label = {0:'patients', 1:'studies', 2:'medicines', 3:'organisation', 4:'covid medicines', 5:'infection', 6:'treatment'}
    npr['Topic Label'] = npr['Topic'].map(topic_label)
    
    npr = npr.assign(Article=df_name_of_file['PDF Name'])

    npr.to_csv('classified_output.csv')



def extract_size(pdf_path):
    

    for filename in os.listdir(pdf_path):

        if filename.endswith('.pdf'):

            pdf_document = f'{pdf_path}{filename}'
            file_size = os.stat(pdf_document).st_size / (1024 * 1024)

            file_size_list.append(file_size)





# extraction of data from pdf
extract_information('Covid_medicine/')


# extraction of data from pdf
# extract_size('Covid_medicine/')
