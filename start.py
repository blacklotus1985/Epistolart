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
import smooth_inverse

new = {
"Mittente": "Averardo Medici di Castellina-Curzio Picchena-11/08/1624-20214",
"Destinatario": "Francesco",
"Luogo di spedizione":"Firenze",
"Giorno di spedizione":28,
"Mese di spedizione":9,
"Anno di spedizione":1570,
"Ricerca libera":"...] Preso che V.A avere per se un paramento che io il mandare col suo letto et uno studiolo non il paio grave di donare al Sig.r Duca Ferdinando I Gonzaga per mio parte un tavolino col suo piede et quanto a una cassa di greco che con il suddetto roba s' inviare per pareggiare il soma ...]"}

begin = datetime.now()
print(begin)
text_of_letter = new["Ricerca libera"]
#json_new = json.dumps(new) #### ATTENZIONE ricordarsi funzione per trasformare json in dizionario qua fingo di avere gia dizionario

#dict_first_letter,text_of_letter = smooth_inverse.json_to_dict(new, text="Ricerca libera")

conf,main_path,path,ft,df_read,tagger,graph = smooth_inverse.get_parameters()

df_read,cleaned_corpus,combined_index,new_letter_cleaned = smooth_inverse.cleaning(df_read=df_read,conf=conf,text=text_of_letter,main_path=main_path
                                                                                   ,tagger=tagger)

vocab_dict = smooth_inverse.get_dict(text_of_letter=new_letter_cleaned)

df_tf_idf = smooth_inverse.calculate_cosine_similarity(cleaned_corpus=cleaned_corpus,dict_text=vocab_dict,combined_index=combined_index)

df_cosine = smooth_inverse.calculate_letters_similarity(model=ft,df_tf_idf=df_tf_idf,df_read=df_read,constant=0,new_text=new_letter_cleaned,letter_name="new_letter",
                                            conf=conf,sort=True)

df_cosine.to_excel(os.getcwd() + conf.get("OUTPUT", "average_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")

end = datetime.now()
print(end)
print("finished algorithm")

