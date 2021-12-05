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
"Mittente": "'Giorgio Vasari-Francesco Leoni-09/08/1544-568'",
"Destinatario": "Francesco",
"Luogo di spedizione":"Firenze",
"Giorno di spedizione":28,
"Mese di spedizione":9,
"Anno di spedizione":1570,
"Ricerca_libera":"Mercoledì in una Congregazione de Agostiniani, che fù tenuta avanti S. B. [Alexander VII] fù stabilita la canonizazione d Beato Tomaso di VIllanova Agostiniano che dovrà farsi il primo giorno di novembre prossimo a spese della Religione. Discorrendosi della compra di Farnese pare che ha stata fatta à caro prezzo essendosi calculata à uno e mezzo percento con qualche dubbio d'intoppi, ma non è mancato che hà detto che meno so considero questo, che il gusto di haver quel luogo della Casa Farnese in ricompensa di quello [che] tiene la medesima della Casa Chigi [...] La Regina [Kristina Wasa] è ben guarita, et fù giovedì a vedere il giardino di S. Pietro et mentre è stata in letto N. S. ha mandato il suo medico [Matteo] Naldi a servirla continuamente, et nel medesimo tempo si è accrescuito un'altro Corpo di Guardia nella piazza di Monte Cavallo [...] Sono arrivati qui da Livorno dui Inglesi che dicono esser stati a Constantinapoli, et si fanno conoscere si Setta Tremolanti: Dicono che lo Spirito Santo li suggerisce cio che devono fare, et dire, et che son fuora con fine di corregiere li errori del Mondo. Hanno chiesto con gran premura audienza dal Papa, ma son stati messi in prigione, dove dicono  grandissime heresia, et moltissime pazzie mostrandosi più tosto pazzi, che furbi; parlano però lingua latina benche con qualche rozzezza, et mostrano di haver cognitione di lettere. Iermattina si tenne l'esame de' Vescovi dove tra quattro soggetti passò un Spagnolo per il Vescovato di Patti, et il Padre Mei Lucchese per Bisignano portato da favori della Signora Leonora Musica.  Ci son qui molti ammalati di catarri e infreddature, che si liberano con una o dui febbri. Il Signore Ammiraglio Sergardi si trova in letto per tale accidente et doppo essersi cavato sangue si è liberato ma non esce per ancora […]"}
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

def connect_db(conf,item="Paragraph"):
    """
    connect to db
    :param conf: configuration parameters
    :return: graph and dataframe
    """
    graph = connection.connect(conf)
    df_read = word2vec.graph_to_pandas(graph,query=item)

    return graph,df_read



def get_parameters(load, item="Paragraph"):
    """
    get parameters required for algorithm
    :return: configuration, main path of project, path of this file, fast text  w2vec model, pickled file of database, treetagger model, graph db connection
    """
    conf = connection.get_conf()
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it", TAGDIR="/home/alex/Scaricati")
    if not load:
        df_read = pickle.load(open(os.getcwd() + "/data/df_read.fourth", "rb"))
        graph = connection.connect(conf)
    else:
        graph,df_read = connect_db(conf,item)
    return conf,main_path,path,ft,df_read,tagger,graph


def cleaning(df_read,conf,text,main_path,tagger):
    """
    cleaning of letter recieved by user and creating new index to find letter_paragraph
    :param df_read: pandas dataframe of paragraphs
    :param conf: configuration file
    :param text: text to clean
    :return: dataframe cleaned and filtered with added column of letter_paragraph, corpus cleaned, list of letter_paragraph names
    """
    df_read = df_read[df_read['translation'].notna()]
    df_read = df_read[df_read['translation'].map(len) > conf.getint("ITEMS","min_len")]
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
    return df_read,cleaned_corpus,combined_index,new_letter_cleaned

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



def get_dict(text_of_letter):
    """
    creates vocabulary for tf idf calculus
    :param text_of_letter: text of letter of user
    :return: dictionary for tf idf
    """
    dict_text = word2vec.create_vocabulary(text_of_letter)
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

def calculate_letters_similarity(model,df_tf_idf,df_read,constant,new_text,letter_name,conf,sort=True):
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
    for elem in index_old_paragraphs:
        df_word_vector,mean_vector = create_document_vector(paragraph=df_read.loc[df_read.index[counter],conf.get("ITEMS","text_field")],
                                                            paragraph_id=elem,model=model,df_tf_idf=df_tf_idf,constant=constant)
        dict = {"paragraph_name":elem,"mean_vector":list(mean_vector)}
        dict_cosine_list.append(dict)
        counter = counter+1
    dict_cosine_list.insert(0,new_dict)
    big_df = pd.DataFrame(dict_cosine_list)
    big_df_new = pd.DataFrame(big_df['mean_vector'].to_list(), index=big_df.paragraph_name)
    indexes = list(big_df_new.index)
    cosine_sim = np.round(cosine_similarity(big_df_new, big_df_new), 8)
    df_cosine = pd.DataFrame(cosine_sim, index=indexes, columns=[indexes])
    if sort:
        df_cosine = df_cosine.T
    df_cosine = df_cosine.sort_values("new_letter", ascending=False)
    df_cosine = df_cosine.head(conf.getint("ITEMS","records"))
    return df_cosine

def create_test_df(df_cosine,df_read,graph,query="Letter"):
    start = datetime.now()
    print ("start load db at {}".format(start))

    db_letter = word2vec.graph_to_pandas(graph,query=query)
    start = datetime.now()
    print("end load db at {}".format(start))
    df_cosine = df_cosine.iloc[1:, :]
    df_cosine.loc[df_cosine['new_letter'] >0]
    similarity_values = df_cosine.iloc[:,0].values
    counter = 0
    dict_list = []
    for elem in df_cosine.index:
        id_composed = " ".join(elem)
        id_letter = id_composed.split("---")[0]
        id_paragraph = id_composed.split("---")[1]
        text_letter = db_letter[db_letter.letter_id==id_letter]
        text_letter = text_letter.transcription.values[0]
        parag_text = df_read[df_read.let_par == df_cosine.index[counter][0]]
        parag_text = parag_text.text.values[0]
        dict = {"letter_name":id_letter,"id_paragraph":id_paragraph,"text_letter":text_letter,"text_paragraph":parag_text,"similarity_value":similarity_values[counter]}
        counter = counter +1
        dict_list.append(dict)
    df_final = pd.DataFrame(dict_list)
    return df_final


