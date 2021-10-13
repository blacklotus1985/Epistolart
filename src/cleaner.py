import re
import treetaggerwrapper
import corrector
from src import connection
conf = connection.get_conf()


def add_stopwords(file,stopwords):
    """
    adds stopwords from txt file
    :param file: name of file
    :param stopwords: stopword list
    :return: extended stopword list
    """
    with open(file) as f:
        contents = f.read()
        contents = contents.splitlines()
        stopwords.extend(contents)
        return stopwords

def removeNonAlpha(text):
    """
    removes non alphabetical characthers
    :param text: text to clean
    :return: cleaned text
    """
    text = re.sub("[^a-zA-Z0-9]+", " ",text)
    text = re.sub("#\S+", " ", text)
    text = re.sub("@\S+", " ", text)
    text = text.lower()
    return text


def removeStopWords(text,conf,stopwords,remove_short_words=True):
    """
    removes stop words in stopwords
    :param text: text to analyze
    :param stopwords: stopword list
    :param remove_short_words: removes words smaller than 2 characthers
    :return: cleaned text
    """
    words = text.split()
    if remove_short_words:
        words = [i for i in words if len(i) > conf.getint("ITEMS","min_len_words")]
    words = [word for word in words if not word in set(stopwords)]
    text = ' '.join(words)
    return text

def lemmatize(text,tagger):
    """
    lemmatizes in italian language with treetagger
    :param text: text to analyze
    :param tagger: tree tagger object
    :return: cleaned text
    """
    tags = tagger.tag_text(text)
    tags = treetaggerwrapper.make_tags(tags)
    cleaned_text = []
    for elem in tags:
        lemma = elem.lemma
        lemma = re.sub(r'\w+\|\b', '', lemma)
        cleaned_text.append(lemma)
    text = ' '.join(cleaned_text)
    return text


def clean_text(df,conf,stopwords,tagger,column='testo'):
    """
    clean dataframe of letters
    :param df: dataframe with metadata
    :param column: column to clean
    :return: dataframe cleaned
    """
    cleaned_corpus = []
    for elem in df[column]:
        elem = removeNonAlpha(elem)
        '''
        try:
            elem = corrector.correct_letter(elem,debug=False)
            print ("call made")
        except:
            print("couldn't make call")
        '''
        #elem = lemmatize(elem,tagger)
        elem = removeStopWords(elem,conf=conf,stopwords=stopwords)
        cleaned_corpus.append(elem)
    return cleaned_corpus






