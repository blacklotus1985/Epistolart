# importing the requests library
import requests
from datetime import datetime
def correct_letter(text,URL="http://epistolarita-develop.kube.simultech.it/spellcheck", debug = True, prt= False):
    """
    corrects letter using spellchecker
    :param text: text to correct
    :param URL: url of post call
    :return: corrected text
    """
    dict  = {"transcription":text}
    if prt:
        start = datetime.now()
        print("start at {}".format(start))
    if debug:
        print("correct_letter_called")
    if not isinstance(dict["transcription"],str):
        dict["transcription"]=" ".join(dict["transcription"])
        response = requests.post(url=URL, json=dict)
        if debug:
            print ("non string request worked")
        if prt:
            start = datetime.now()
            print("start at {}".format(start))
    else:
        response = requests.post(url=URL, json=dict)
        if debug:
            print("string request worked")
        if prt:
            start = datetime.now()
            print("start at {}".format(start))
    return response.json()['translation']




