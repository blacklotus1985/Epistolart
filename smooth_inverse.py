# Importing necessary libraries
import pandas as pd
import numpy as np
import fasttext.util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
from src import connection
import treetaggerwrapper
from src import cleaner
from stop_words import get_stop_words
import converter
from sklearn.decomposition import PCA
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import pickle
import word2vec
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

new = {
"Mittente": "Averardo Medici di Castellina-Curzio Picchena-11/08/1624-20214",
"Destinatario": "Francesco",
"Luogo di spedizione":"Firenze",
"Giorno di spedizione":28,
"Mese di spedizione":9,
"Anno di spedizione":1570,
"Ricerca_libera":"...] Preso che V.A avere per se un paramento che io il mandare col suo letto et uno studiolo non il paio grave di donare al Sig.r Duca Ferdinando I Gonzaga per mio parte un tavolino col suo piede et quanto a una cassa di greco che con il suddetto roba s' inviare per pareggiare il soma ...]"}
json_new = json.dumps(new)

dict_first_letter = json.loads(json_new)

def json_to_dict(json_letter,text = "Ricerca_libera"):
    """
    transform json into dictionary
    :param json_letter: new letter from user
    :param text: field of json in which text of letter is kept
    :return: dictionary of json and the text of the letter
    """
    dict_first_letter = json.dumps(json_letter)
    return dict_first_letter,dict_first_letter[text]

def get_parameters():
    """
    get parameters required for algorithm
    :return: configuration, main path of project, path of this file, fast text  w2vec model, pickled file of database, treetagger model, graph db connection
    """
    conf = connection.get_conf()
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    df_read = pickle.load(open(os.getcwd() + "/data/df_read.fourth", "rb"))
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    graph = connection.connect(conf)
    return conf,main_path,path,ft,df_read,tagger,graph


def cleaning(df_read,conf,text):
    """
    cleaning of letter recieved by user and creating new index to find letter_paragraph
    :param df_read: pandas dataframe of paragraphs
    :param conf: configuration file
    :param text: text to clean
    :return: dataframe cleaned and filtered with added column of letter_paragraph, corpus cleaned, list of letter_paragraph names
    """
    df_read = df_read[df_read['translation'].notna()]
    df_read = df_read[df_read['translation'].map(len) > conf.getInt("INPUT","min_len")]
    stopwords = get_stop_words('it')
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    cleaned_corpus = cleaner.clean_text(df_read, conf, stopwords=stopwords, tagger=tagger, column='translation')
    new_letter_cleaned = cleaner.clean_new_letter(letter=text, conf=conf, stopwords=stopwords)
    cleaned_corpus.append(new_letter_cleaned)
    df_read['let_par'] = [x + '---' + y for x, y in zip(df_read.letter_id.values, df_read.name.values)]
    combined_index = list(df_read['let_par'].values)
    combined_index.append("new_letter")
    return df_read,cleaned_corpus,combined_index

def create_document_vector(paragraph,paragraph_id,model,df_tf_idf,constant):
    """
    creates document embedding weighted with tf idf
    :param paragraph: text to analyze
    :param paragraph_id: paragraph id in the df_tf_idf matrix
    :param model: fast text model
    :param df_tf_idf: dataframe of tf_idf
    :param constant: costant to multiply non existing words (default is 0)
    :return: dataframe of each word in text with w2vec embedding, mean vector representation of document
    """
    list = []
    list_paragraph = paragraph.split(" ")
    for word in list_paragraph:
        if word in df_tf_idf.columns:
            tf_value = df_tf_idf.loc[paragraph_id,word]
            dict = {"word": word, "vector": model.get_word_vector(word)*tf_value}
        else:
            dict = {"word": word, "vector": model.get_word_vector(word) * constant}
        list.append(dict["vector"])
    df_word_vector = pd.DataFrame(list,index=list_paragraph)
    mean_vector = df_word_vector.mean().values
    return df_word_vector,mean_vector

def create_new_letter_vector(text,letter_name,model,df_tf_idf):
    """
    creates the vector of the new letter passed by user
    :param text: text passed by user
    :param letter_name: letter name passed by user, by default called "new_letter"
    :param model: fast text model
    :param df_tf_idf: dataframe of tf_idf
    :return:
    """
    list = []
    list_paragraph = text.split(" ")
    for word in list_paragraph:
        tf_value = df_tf_idf.loc[letter_name, word]
        dict = {"word": word, "vector": model.get_word_vector(word) * tf_value}
        list.append(dict["vector"])
    df_word_vector = pd.DataFrame(list, index=list_paragraph)
    mean_vector = df_word_vector.mean().values
    return df_word_vector, mean_vector



def get_dict():
    dict_text = word2vec.create_vocabulary(dict_first_letter["Ricerca libera"],dict_first_letter)
    return dict_text

def calculate_cosine_similarity(cleaned_corpus,dict_text,combined_index):
    """
    calculates tf_idf of all documents in database and the new letter from user
    :param cleaned_corpus: list of text letters with also the user letter
    :param dict_text: vocabulary to use for tf_idf based on user's letter
    :param combined_index: composed index to identify letter---paragraph
    :return: dataframe of tf_idf
    """
    df_tf_idf, raw_matrix = word2vec.calculate_tf_idf(corpus=cleaned_corpus, vocabulary=dict_text,index=combined_index)
    return df_tf_idf

def calculate_similarity(model,df_tf_idf,df_read,constant,new_text,letter_name,conf,sort=True,print=True):
    """
    calculates similarity of all letters with the user's letter
    :param model: fast text model
    :param df_tf_idf: dataframe of tf_idf
    :param df_read: dataframe of database letters
    :param constant: costant to multiply non existing words (default is 0)
    :param new_text: new letter of user
    :param letter_name: letter name of user
    :param conf: configuration file
    :param sort: boolean to achieve or not sorted results
    :param print: boolean to print time of running code
    :return: dataframe of cosine similarities
    """
    index_paragraph_list = list(df_tf_idf.index)
    index_old_paragraphs = index_paragraph_list[:-1]
    df_word_vector_new_letter,mean_vector_new_letter = create_new_letter_vector(text=new_text,letter_name=letter_name,model=model,df_tf_idf=df_tf_idf)
    new_dict = {"paragraph_name":"new_letter","mean_vector":list(mean_vector_new_letter)}
    dict_cosine_list = []
    counter = 0
    if print:
        start = datetime.now()
        print("start cosine at {}".format(start))
    for elem in index_old_paragraphs:
        df_word_vector,mean_vector = create_document_vector(paragraph=df_read.loc[df_read.index[counter],conf.get("INPUT","translation")],
                                                            paragraph_id=elem,model=model,df_tf_idf=df_tf_idf,constant=constant)
        dict = {"paragraph_name":elem,"mean_vector":list(mean_vector)}
        dict_cosine_list.append(dict)
        counter = counter+1
    if print:
        start = datetime.now()
        print("end cosine at {}".format(start))
    dict_cosine_list.insert(0,new_dict)
    big_df = pd.DataFrame(dict_cosine_list)
    big_df_new = pd.DataFrame(big_df['mean_vector'].to_list(), index=big_df.paragraph_name)
    indexes = list(big_df_new.index)
    cosine_sim = np.round(cosine_similarity(big_df_new, big_df_new), 8)
    df_cosine = pd.DataFrame(cosine_sim, index=indexes, columns=[indexes])
    if sort:
        df_cosine = df_cosine.T
    df_cosine = df_cosine.sort_values("new_letter", ascending=False)
    df_cosine = df_cosine.head(conf.get("INPUT","records"))
    return df_cosine


if __name__ == '__main__':
    begin = datetime.now()
    print ("algorithm started at {}".format(begin))
    conf = connection.get_conf()
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    df_read = pickle.load(open(os.getcwd()+"/data/df_read.fourth", "rb"))
    # Dump the variable tz into file save.p

    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    graph = connection.connect(conf)
    #df_read = word2vec.graph_to_pandas(graph,query="Paragraph")
    #pickle.dump(df_read, open(os.getcwd() + "/data/df_read.fourth", "wb"))
    #print('finished dump ###@@@@###')
    print(df_read.shape)
    #df_read = pickle.load(open(os.getcwd()+"/data/df_read.second", "rb"))
    print(df_read.shape)
    #testo = conf.get("ITEMS","testo")
    df_read = df_read[df_read['translation'].notna()]
    df_read = df_read[df_read['translation'].map(len) > 600]
    #df_read = df_read.sample(16000)
    end = datetime.now()
    print(df_read.shape)
    # df_read = pickle.load(open(os.getcwd()+"/data/df_read.second", "rb"))
    print(df_read.shape)
    stopwords = get_stop_words('it')
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    cleaned_corpus = cleaner.clean_text(df_read, conf, stopwords=stopwords, tagger=tagger, column='translation')
    new_letter_cleaned = cleaner.clean_new_letter(letter=dict_first_letter["Ricerca libera"],conf=conf,stopwords=stopwords)
    cleaned_corpus.append(new_letter_cleaned)
    #df_read = df_read.append(dict_first_letter, ignore_index=True)
    #dict = {"translation": new_letter_cleaned, "letter_id": 100000, "name": "new_letter"}
    #df_read = df_read.append(dict, ignore_index=True)
    end = datetime.now()
    print("end cleaned all stopwords at {}".format(end))
    # text_testing = cleaned_corpus[3]
    #words = 'Statua Arte Dipinto Papa Amore Colore Ritratto'
    #list_words = words.split(" ")
    # text_testing = get_neighbors(list_words, ft=ft, k=10,tuple=True)
    print("before cleaning 2")
    # print(text_testing)
    # cleaned_corpus[3] = text_testing
    # cleaned_corpus[3] = cleaner.removeNonAlpha(cleaned_corpus[3])
    # cleaned_corpus[3] = cleaner.removeStopWords(cleaned_corpus[3],conf,stopwords=stopwords,remove_short_words=False)
    # cleaned_corpus[3] = cleaner.lemmatize(cleaned_corpus[3],tagger=ft)
    # text_testing = get_neighbors(text_testing,ft=ft,k=5)
    dict_text = word2vec.create_vocabulary(dict_first_letter["Ricerca libera"])
    df_read['let_par'] = [x + '---' + y for x, y in zip(df_read.letter_id.values, df_read.name.values)]
    combined_index = list(df_read['let_par'].values)
    combined_index.append("new_letter")


    # tempo di computazione tipo nullo
    df_tf_idf, raw_matrix = word2vec.calculate_tf_idf(corpus=cleaned_corpus, vocabulary=dict_text,
                                             index=combined_index)  # remember to fix index=row_id when you find id letter in paragraphs
    end = datetime.now()
    print("start mean vector at {}".format(end))
    df_final = calculate_similarity(paragraph=new["Ricerca libera"],model=ft,df_tf_idf=df_tf_idf,df_read=df_read,constant=0)
    df_final.to_excel(os.getcwd() + conf.get("OUTPUT", "average_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")

    #df_paragraph, mean_vector = create_document_vector(paragraph=df_read.iloc[1,4],paragraph_id=df_tf_idf.index[4],model=ft,df_tf_idf=df_tf_idf,constant=0)
    end = datetime.now()
    print("end mean vector at {}".format(end))
    start = datetime.now()
