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
"Mittente": "Alex",
"Destinatario": "Francesco",
"Luogo di spedizione":"Firenze",
"Giorno di spedizione":28,
"Mese di spedizione":9,
"Anno di spedizione":1570,
"Ricerca libera":"Perché fino adesso non  è stare di bisogno scrìvere, e per non avere lo cuore lietto e iscrivendo altro che cose importante, bisognerebbe scrìvere lamentazione più che quelle di Geremia, per fino adesso non t'ho scrìvere. E perché fa di necessità scrìvere, mi suono messo a farlo. E la causa è che e' nostro Lucca Paganelli ha opposto qui alla mercanzìa, che per essere èssere le ragioni, donde dipende e' débito Aretino, non lo potere stringere di qua, di sorte che avere fatto, come chi zappa i[n] rena, che quanto più zappa, manco lavorare e basta. "
}

json_new = json.dumps(new)

dict_first_letter = json.loads(json_new)

def get_text(new_json,text="Ricerca_libera"):
    dict = json.dumps(new_json)
    return dict[text]

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

def create_new_letter_vector(text,letter_name,model,df_tf_idf):
    list = []
    list_paragraph = text.split(" ")
    for word in list_paragraph:
        tf_value = df_tf_idf.loc[letter_name, word]
        dict = {"word": word, "vector": model.get_word_vector(word) * tf_value}
        list.append(dict["vector"])
    df_word_vector = pd.DataFrame(list, index=list_paragraph)
    mean_vector = df_word_vector.mean().values
    return df_word_vector, mean_vector


def calculate_similarity(paragraph,model,df_tf_idf,df_read,constant):
    index_paragraph_list = list(df_tf_idf.index)
    df_word_vector_new_letter,mean_vector_new_letter = create_new_letter_vector(text=cleaned_corpus[-1],letter_name="new_letter",model=ft,df_tf_idf=df_tf_idf)
    dict_cosine_list = []
    counter = 0
    for elem in index_paragraph_list:
        df_word_vector,mean_vector = create_document_vector(paragraph=df_read.loc[df_read.index[counter],"translation"],paragraph_id=elem,model=model,df_tf_idf=df_tf_idf,constant=constant)
        try:
            cosine_sim_value = 1 - spatial.distance.cosine(mean_vector_new_letter, mean_vector)
            cosine_sim_value = np.round(cosine_sim_value,5)
        except:
            cosine_sim_value = 0
            print("invalid value in float scalars")
            counter = counter+1
            print(counter)
        dict = {"paragraph_name":elem,"cosine_similarity":cosine_sim_value}
        dict_cosine_list.append(dict)
        counter = counter+1
    df_cosine_final = pd.DataFrame(dict_cosine_list)
    print(1)
    return df_cosine_final

def create_new_df_matrix (df_read,dict):
    a=1






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
    new_letter_cleaned = cleaner.clean_new_letter(letter=dict_first_letter["Ricerca libera"],conf=conf,stopwords=stopwords)
    cleaned_corpus.append(new_letter_cleaned)
    #df_read = df_read.append(dict_first_letter, ignore_index=True)
    #dict = {"translation": new_letter_cleaned, "letter_id": 100000, "name": "new_letter"}
    #df_read = df_read.append(dict, ignore_index=True)
    end = datetime.now()
    print("end cleaned all stopwords at {}".format(end))
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
    dict_text = word2vec.create_vocabulary(dict_first_letter["Ricerca libera"])
    letter_name = df_read.iloc[1, 8]
    # print(letter_name)
    # print(cleaned_corpus[3])
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
