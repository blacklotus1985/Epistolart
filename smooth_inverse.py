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

new = {
"Mittente": "Alex",
"Destinatario": "Francesco",
"Luogo di spedizione":"Firenze",
"Giorno di spedizione":28,
"Mese di spedizione":9,
"Anno di spedizione":1570,
"Ricerca libera":"Cras mattis..."
}

json_new = json.dumps(new)

dict = json.loads(json_new)
def create_document_vector(paragraph,paragraph_id,model,df_tf_idf,constant):
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
    testo = conf.get("ITEMS","testo")
    df_read = df_read[df_read['translation'].notna()]
    df_read = df_read[df_read['translation'].map(len) > 400]
    #df_read = df_read.sample(1100)
    end = datetime.now()

    stopwords = get_stop_words('it')
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    cleaned_corpus = cleaner.clean_text(df_read, conf, stopwords=stopwords, tagger=tagger, column='translation')
    end = datetime.now()
    end = datetime.now()
    print("end get stopwords at {}".format(end))
    # text_testing = cleaned_corpus[3]
    words = 'Statua Arte Dipinto Papa Amore Colore Ritratto'
    list_words = words.split(" ")
    # text_testing = get_neighbors(list_words, ft=ft, k=10,tuple=True)
    print("before cleaning 2")
    # print(text_testing)

    # cleaned_corpus[3] = text_testing
    # cleaned_corpus[3] = cleaner.removeNonAlpha(cleaned_corpus[3])
    # cleaned_corpus[3] = cleaner.removeStopWords(cleaned_corpus[3],conf,stopwords=stopwords,remove_short_words=False)
    # cleaned_corpus[3] = cleaner.lemmatize(cleaned_corpus[3],tagger=ft)

    # text_testing = get_neighbors(text_testing,ft=ft,k=5)
    dict_text = word2vec.create_vocabulary(cleaned_corpus[1])
    letter_name = df_read.iloc[1, 8]
    # print(letter_name)
    # print(cleaned_corpus[3])
    df_read['let_par'] = [x + '---' + y for x, y in zip(df_read.letter_id.values, df_read.name.values)]
    combined_index = df_read['let_par'].values


    # tempo di computazione tipo nullo
    df_tf_idf, raw_matrix = word2vec.calculate_tf_idf(corpus=cleaned_corpus, vocabulary=dict_text,
                                             index=combined_index)  # remember to fix index=row_id when you find id letter in paragraphs
    end = datetime.now()
    print("start mean vector at {}".format(end))
    df_paragraph, mean_vector = create_document_vector(paragraph=df_read.iloc[1,4],paragraph_id=df_tf_idf.index[4],model=ft,df_tf_idf=df_tf_idf,constant=0)
    end = datetime.now()
    print("end mean vector at {}".format(end))
    start = datetime.now()
