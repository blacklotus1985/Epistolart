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


def avg_w2vec(tf_idf_matrix,model):
    """
    calculates similarity results using w2vec average and tf idf matrix
    :param tf_idf_matrix: tf idf matrix
    :param model: fast text object
    :return: results of similarities between texts
    """
    words = list(tf_idf_matrix.columns)
    big_list = []
    small_list = []
    for index, row in tf_idf_matrix.iterrows():
        array = row.values
        word_index = 0
        for word in words:
            tf = array[word_index]*100
            if tf:
                vector_word = model.get_word_vector(word)
                avg = np.average(vector_word)*100
                result_avg = np.round((tf * avg),3)
                small_list.append(result_avg)
            else:
                small_list.append(0)
            word_index +=1
        big_list.append(small_list)
        small_list = []
    df_result = pd.DataFrame(big_list,index=tf_idf_matrix.index,columns=tf_idf_matrix.columns)
    df_result = df_result/100
    return df_result




def save_lemmatized_text(df,cleaned_coprus,column_name='testo',save=True):
    """
    save lemmatized text in dataframe
    :param df: starting df with not lemmatized column
    :param cleaned_coprus: lemmatized text
    :param column_name: column name of lemmatized df
    :param save: save to excel
    :return:
    """
    del df[column_name]
    df[column_name] = cleaned_coprus
    if save:
        df.to_excel(os.getcwd()+'/data/df_lemmatized.xlsx',index=False)
    return df

def gens_tf(corpus):
    dct = Dictionary([corpus])  # fit dictionary
    docs = [dct.doc2bow(line) for line in [corpus]]  # convert corpus to BoW format
    model = TfidfModel(docs)  # fit model
    vector = model[docs[0]]  # apply model to the first corpus document
    return model

def calculate_tf_idf(corpus, vocabulary, index, max_df=0.4,min_df = 1,max_features = 10000): # removed rownames as index of matrix cause no id for now
    cv = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features,max_df=max_df,min_df=min_df,vocabulary=vocabulary)
    X = cv.fit_transform(corpus)
    Y = X.toarray()
    count_vect_df = pd.DataFrame(Y, columns=cv.get_feature_names(),index=index)# removed index = rownames
    return count_vect_df,X


def graph_to_pandas(graph,query):
    list = graph.nodes.match(query).all()
    id_list = []
    for node in list:
        id_list.append(node.identity)
    df =  pd.DataFrame(list)
    df['id'] = id_list
    return df

def get_letter_from_paragraph(df,id_paragraph, graph):
    '''
    relationship_matcher = RelationshipMatcher(graph=graph)
    letter_node = relationship_matcher.get(id_paragraph)
    letter_key = letter_node.end_node.get("letter_id")
    '''
    letter_dict = graph.run("MATCH (n:Paragraph)<-[:AS_PARAGRAPH]-(l:Letter) where ID(n) = {0} RETURN n,l".format(id_paragraph))
    data = letter_dict.data()[0]['l']
    letter_key = data.get("letter_id")
    return letter_key


def get_all_letters(graph,df):
    list = [get_letter_from_paragraph(df=df,id_paragraph=id,graph=graph) for id in df['id']]
    df['letter_id'] = list
    return df

def create_vocabulary(text):
    lst = text.split(" ")
    dict = {}
    final_list = list(dict.fromkeys(lst))
    dict = {i: final_list[i] for i in range(0, len(final_list))}
    swap_dict = {value:key for key, value in dict.items()}
    return swap_dict

def get_neighbors(text, ft,k,tuple): # item returned is a string not a list
    text_words = []
    for word in text:
        word_list = ft.get_nearest_neighbors(word,k)
        text_words.append(word_list)
    if not tuple:
        flat_list = [item for sublist in text_words for item in sublist]
    else:
        flat_list = [item[1] for sublist in text_words for item in sublist]
    flat_list = ' '.join(flat_list)
    return flat_list


# Restituisci nodo lettera completo!

if __name__ == '__main__':
    begin = datetime.now()
    print ("algorithm started at {}".format(begin))
    conf = connection.get_conf()
    main_path = os.getcwd()
    path = os.path.dirname(os.getcwd())
    ft = fasttext.load_model(main_path + '/data/cc.it.300.bin')
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
    #graph = connection.connect(conf)
    #df_read = graph_to_pandas(graph,query="Paragraph")
    df_read = pickle.load(open(os.getcwd()+"/data/df_read.second", "rb"))
    print(df_read.shape)
    testo = conf.get("ITEMS","testo")
    df_read = df_read[df_read['translation'].notna()]
    df_read = df_read[df_read['translation'].map(len) > 400]
    #df_read = df_read.sample(1100)
    end = datetime.now()
    rk = False
    if rk:
        end = datetime.now()
        print("start rake at {}".format(end))
        kw = rake.extract_keywords(df_read.loc[3], lang=None)
        end = datetime.now()
        print("end rake at {}".format(end))
    print("start get stopwords at {}".format(end))
    stopwords = get_stop_words('it')
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-aggettivi.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-varie.txt', stopwords=stopwords)
    stopwords = cleaner.add_stopwords(main_path + '/data/stp-verbi.txt', stopwords=stopwords)
    cleaned_corpus = cleaner.clean_text(df_read, conf, stopwords=stopwords, tagger=tagger, column='translation')
    end = datetime.now()
    print("end get stopwords at {}".format(end))
    #text_testing = cleaned_corpus[3]
    words = 'Statua Arte Dipinto Papa Amore Colore Ritratto'
    list_words = words.split(" ")
    #text_testing = get_neighbors(list_words, ft=ft, k=10,tuple=True)
    print ("before cleaning 2")
    #print(text_testing)

    #cleaned_corpus[3] = text_testing
    #cleaned_corpus[3] = cleaner.removeNonAlpha(cleaned_corpus[3])
    #cleaned_corpus[3] = cleaner.removeStopWords(cleaned_corpus[3],conf,stopwords=stopwords,remove_short_words=False)
    #cleaned_corpus[3] = cleaner.lemmatize(cleaned_corpus[3],tagger=ft)

    #text_testing = get_neighbors(text_testing,ft=ft,k=5)
    dict_text = create_vocabulary(cleaned_corpus[3])
    letter_name = df_read.iloc[3, 8]
    #print(letter_name)
    #print(cleaned_corpus[3])
    df_read['let_par'] = [x + '---' + y for x, y in zip(df_read.letter_id.values, df_read.name.values)]
    combined_index = df_read['let_par'].values

    # Dump the variable tz into file save.p
    #pickle.dump(df_read, open(os.getcwd()+"/data/df_read.second", "wb"))
    print('finished dump ###@@@@###')
    # tempo di computazione tipo nullo
    df_tf_idf, raw_matrix = calculate_tf_idf(corpus=cleaned_corpus, vocabulary = dict_text, index=combined_index) # remember to fix index=row_id when you find id letter in paragraphs
    start = datetime.now()
    print("start calculate dataframe w2vec at {}".format(start))
    converted_df = converter.calculate_dataframe(df_tf_idf,model=ft)
    total_big_df = converter.total_w2vec(converted_df, ft, df_tf_idf)
    start = datetime.now()
    print("end calculate dataframe w2vec at {}".format(start))
    start = datetime.now()
    #print ("start saving total df at {}".format(start))
    #total_big_df.to_csv(os.getcwd() + conf.get("OUTPUT", "total_big_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".csv")
    start = datetime.now()
    #print("finished saving total df at {}".format(start))
    #final_result = avg_w2vec(df_tf_idf,model=ft)
    red = False
    if red:
        pc = PCA(n_components="mle")
        matrix = pc.fit_transform(total_big_df)
    start = datetime.now()
    print("before cosine sim {}".format(start))
    cosine_sim = np.round(cosine_similarity(total_big_df, total_big_df),15)
    start = datetime.now()
    print("after cosine sim {}".format(start))
    df_cosine = pd.DataFrame(cosine_sim,index=combined_index,columns=[combined_index])
    try:# add index=row_id,columns=[row_id] when fixed id letter problem
        df_cosine = df_cosine.sort_values(letter_name,ascending=False)
    except:
        print("cannot sort df cosine")
        pass
    start = datetime.now()
    print("start saving bigw2vec at {}".format(start))
    df_cosine.to_excel(os.getcwd()+conf.get("OUTPUT","bigw2vec")+datetime.now().strftime("%d-%m-%y-%H-%M-%S")+".xlsx")

    start = datetime.now()
    print("finished saving bigw2vec at {}".format(start))
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
    #best_df.to_excel(os.getcwd() + conf.get("OUTPUT", "best_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")

    print(datetime.now() - begin)
