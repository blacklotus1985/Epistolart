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

dict_first_letter,text_of_letter = smooth_inverse.json_to_dict(new)
