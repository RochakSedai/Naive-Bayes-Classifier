import fitz
import pandas as pd
import os
import sys


def get_path():
    final_path = []
    path1 = input('Enter path for AI files: ')
    print('Path registered successfully')
    path2 = input('Enter path for WEB files: ')
    print('Path registered successfully')
    final_path.append(path1)
    final_path.append(path2)
    return final_path


def get_final_dataframe(path, flag):
    df = pd.DataFrame(columns = ['Text', 'Label'])
    content = []
    label = []
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            doc = fitz.open(path+'/'+file)
            content_temp=''
            for page in range(len(doc)):
                content_temp = content_temp+doc[page].get_text()
                # print(content_temp)
            content.append(content_temp)
    
    df['Text'] = content
    df['Label'] = flag
    # print(df)
    return df
    
def get_content_of_pdfs(file_path):
    for path in file_path:
        if '\\AI' in path:
            print('Heres in AI path')
            print(path)
            df_ai = get_final_dataframe(path,1)
            # sys.exit()
        elif '\\WEB' in path:
            print('Here is WEB path')
            print(path)
            df_web = get_final_dataframe(path,0)
            
    df = df_ai.append(df_web)
    return df
    
def get_content(file_path):
    df = pd.DataFrame(columns = ['Text', 'Label'])
    df = get_content_of_pdfs(file_path)
    return df

def dataset_generate():
    file_path = get_path()
    dataset=get_content(file_path)
    dataset.to_csv('Dataset.csv')
    # print('Hello World')

if __name__=='__main__':
    dataset_generate()