import os
import pandas as pd
import numpy as np
from src import connection
from collections import Counter


main_path = os.getcwd()
def calculate_dataframe(tf_idf_matrix,model):
    dict_list = []
    for word in tf_idf_matrix.columns:
        dict = {"word":word,"value":model.get_word_vector(word)}
        dict_list.append(dict)
    df = pd.DataFrame(dict_list)
    final_df = pd.DataFrame(df['value'].to_list(), index=df["word"])
    return final_df

def calculate_vec(converted_df,ft,tf_idf_matrix,column_df):
    dict_list = []
    names_col_list = []
    all_words = tf_idf_matrix.columns.to_list()
    for word in all_words:
        tf_value = tf_idf_matrix.loc[column_df,word]
        if tf_value > 0:
            dict = {"word":word, "value":ft.get_word_vector(word)*tf_value}
            dict_list.append(dict['value'])
        else:
            dict_list.append(np.zeros(converted_df.shape[1]))
        vector = list(np.arange(300))
        single_name_list = [word + str(s) for s in vector]
        names_col_list.append(single_name_list)
    flat_list = [item for sublist in dict_list for item in sublist]
    total_name_list = [item for sublist in names_col_list for item in sublist]
    dict = {"title":column_df,"value":flat_list}
    #df = pd.DataFrame(flat_list,index=total_name_list,columns=[column_df])
    return dict,total_name_list

def total_w2vec(converted_df,ft,tf_idf_matrix):
    dict_list = []
    counter = 0
    total_name_list = []
    t = tf_idf_matrix.index.to_list()
    dict = {}
    for letter_title in t:
            del dict
            del total_name_list
            dict,total_name_list = calculate_vec(converted_df, ft, tf_idf_matrix, column_df=letter_title)
            dict_list.append(dict["value"])
            print(counter+1)
            counter = counter + 1
    total_df = pd.DataFrame(dict_list,index = t,columns=total_name_list)
    print (total_df.shape)
    return total_df



def change_same_column(df):
    names = Counter(df.index)
    counter = 1
    df = df.reset_index()
    for i in range (df.shape[0]):
            value = names[df.loc[i,"index"]]
            df.loc[i,"index"] = df.loc[i,"index"] + "_"+ str(counter)
            if not value == counter:
                counter = counter +1
            else:
                counter = 1
    df = df.set_index("index")
    return df



