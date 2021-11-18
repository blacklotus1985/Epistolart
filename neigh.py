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
import rake
import word2vec


begin = datetime.now()
print ("algorithm started at {}".format(begin))
conf = connection.get_conf()
main_path = os.getcwd()
path = os.path.dirname(os.getcwd())
ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
#graph = connection.connect(conf)
#df_read = graph_to_pandas(graph,query="Paragraph")
df_read = pickle.load(open(os.getcwd()+"/data/df_read.p", "rb"))
testo = conf.get("ITEMS","testo")
df_read = df_read[df_read['translation'].notna()]
print ("df_read rows not null {}".format(df_read.shape[0]))
df_read = df_read[df_read['translation'].map(len) > 400]
end = datetime.now()
print("start get stopwords at {}".format(end))
stopwords = get_stop_words('it')
stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
cleaned_corpus = cleaner.clean_text(df_read, conf, stopwords=stopwords, tagger=tagger, column='translation')
end = datetime.now()
print("end get stopwords at {}".format(end))
start = datetime.now()
print ("Length of letter to analyze is  {}".format(len(cleaned_corpus[3])))
print("start calculate neighbors at {}".format(start))
text_testing = word2vec.get_neighbors(cleaned_corpus[3], ft=ft, k=10,tuple=False)
start = datetime.now()
print("end calculate neighbors at {}".format(start))
dict_text = word2vec.create_vocabulary(text_testing)
letter_name = df_read.iloc[3, 8]
df_read['let_par'] = [x + '---' + y for x, y in zip(df_read.letter_id.values, df_read.name.values)]
combined_index = df_read['let_par'].values
df_tf_idf, raw_matrix = word2vec.calculate_tf_idf(corpus=cleaned_corpus, vocabulary=dict_text,index=combined_index)  # remember to fix index=row_id when you find id letter in paragraphs
start = datetime.now()
print("before cosine sim {}".format(start))
cosine_sim = np.round(cosine_similarity(df_tf_idf, df_tf_idf),15)
start = datetime.now()
print("after cosine sim {}".format(start))

start = datetime.now()
print("before save df cosine {}".format(start))
df_cosine = pd.DataFrame(cosine_sim,index=combined_index,columns=[combined_index])
df_cosine.to_excel(os.getcwd()+conf.get("OUTPUT","bigw2vec")+datetime.now().strftime("%d-%m-%y-%H-%M-%S")+".xlsx")
start = datetime.now()
print("after save df cosine {}".format(start))
a = df_cosine.to_numpy().flatten()
b = [x for x in a if x < 0.85]
b = np.array(b)
sort_index_array = np.argsort(b)
sorted_array = b[sort_index_array]
rslt = sorted_array[-20:]
rslt = rslt[1::2]
dict_list = []
for elem in rslt:
    row, column = np.where(df_cosine == elem)
    ind = list(zip(df_cosine.index[row], df_cosine.columns[column]))
    dict = {"letters":ind, "value":elem}
    dict_list.append(dict)

best_df = pd.DataFrame(dict_list,columns=['letters','value'])
start = datetime.now()
print("before save best df {}".format(start))
best_df.to_excel(os.getcwd() + conf.get("OUTPUT", "best_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")
start = datetime.now()
print("after save best df {}".format(start))

print(datetime.now() - begin)
