# Importing necessary libraries
import os
from datetime import datetime
import smooth_inverse
import word2vec
import flask
from flask import jsonify
import logging
"""
Gianluca test keywords
1) amore desiderio passione cupidigia brama
2) sdegno ira rabbia collera disprezzo
3) coraggio osare rischiare ardimento temerario
4) onore gloria fama patria fedeltà 
5) amore passione onore gloria fama
6) sdegno ira delusione smarrimento illusione
7) orgoglio vergogna rabbia amarezza delusione
"""

def main():
    smooth_inverse.create_log()
    # json ricevuto da Fra
    new = {
    "Mittente": "Averardo Medici di Castellina-Curzio Picchena-11/08/1624-20214",
    "Destinatario": "Francesco",
    "Luogo di spedizione":"Firenze",
    "Giorno di spedizione":28,
    "Mese di spedizione":9,
    "Anno di spedizione":1570,
    "Ricerca_libera":"Amore Adorazione Affezione Simpatia Attrazione Cura Tenerezza Compassione Desiderio Passione Infatuazione Brama Speranza Pietà"}
    begin = datetime.now()
    logging.info("algorithm started at {}".format(begin))
    text_of_letter = new["Ricerca_libera"]

    string_end = 150

    text_title = new["Ricerca_libera"]
    text_title = text_title[0:string_end]
    #json_new = json.dumps(new) #### ATTENZIONE ricordarsi funzione per trasformare json in dizionario qua fingo di avere gia dizionario

    #dict_first_letter,text_of_letter = smooth_inverse.json_to_dict(new, text="Ricerca libera")

    conf,main_path,path,ft,df_read,tagger,graph = smooth_inverse.get_parameters(load=True,item="Paragraph")


    df_read = df_read[df_read.letter_id.str.startswith("Francesco Guicciardini")]


    text_of_letter = word2vec.get_neighbors(text_of_letter.split(" "),ft=ft,k=conf.getint("ITEMS","neighbors"),tuple=True)


    df_read,cleaned_corpus,combined_index,new_letter_cleaned = smooth_inverse.cleaning(df_read=df_read,conf=conf,text=text_of_letter,main_path=main_path
                                                                                 ,tagger=tagger)

    vocab_dict = smooth_inverse.get_dict(text_of_letter=new_letter_cleaned)

    df_tf_idf = smooth_inverse.calculate_cosine_similarity(cleaned_corpus=cleaned_corpus,dict_text=vocab_dict,combined_index=combined_index)

    df_cosine = smooth_inverse.calculate_letters_similarity(model=ft,df_tf_idf=df_tf_idf,df_read=df_read,constant=0,new_text=new_letter_cleaned,letter_name="new_letter",
                                                conf=conf,sort=True)

    df_cosine.to_excel(os.getcwd() + conf.get("OUTPUT", "average_df") + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")

    #df_final_for_test = smooth_inverse.create_test_df(df_cosine=df_cosine,df_read=df_read,graph=graph,query="Letter")

    #df_final_for_test.to_excel(os.getcwd() + "/test/" + text_title + " " + str(conf.getint("ITEMS","neighbors"))+"neighbors "  + datetime.now().strftime("%d-%m-%y-%H-%M-%S") + ".xlsx")
    smooth_inverse.response_return(df_cosine, graph, conf=conf, query="Letter")
    end = datetime.now()
    logging.info("end of algorithm at {}".format(end))
    final_time = end-begin
    logging.info("time to run algorithm is {}".format(final_time))
    #return jsonify({"message":"Error password or user not match"})

main()

