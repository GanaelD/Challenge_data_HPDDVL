#https://maelfabien.github.io/machinelearning/NLPfr/#1-tokenisation

import pandas as pd, nltk, string, spacy
from nltk.corpus import stopwords

#python -m nltk.downloader all
nltk.download("stopwords")

def return_token(sentence):
    #print(sentence)
    doc=nltk.word_tokenize(sentence, language='french')
    return [X for X in doc]


def treat_data(input_data):
    other_stopwords = ["alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mien", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "ou", "où", "par", "parce", "pas", "peut", "peu", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez"    , "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être", "l'on"]

    allStopwords = [x for x in other_stopwords]
    for x in stopwords.words("french"):
        allStopwords.append(x)

    #remove duplicates and add punctuation
    allStopwords.extend(string.punctuation)
    allStopwords.append("...")
    allStopwords = list(dict.fromkeys(allStopwords))
    
    
    captions_train = [return_token(line) for line in input_data['Caption']]
    
    captions_train_cleaned = []
    for message in captions_train:
        captions_train_cleaned.append([word.lower() for word in message if word not in allStopwords])
    
    #python3 -m spacy download fr_core_news_md
    nlp = spacy.load('fr_core_news_md')
    
    captions_train_cleaned_stemmed_and_lemmatized_by_spacy = []

    for sentence in captions_train_cleaned:
        #nlp on sentences (create sentence back from list of words separated by " ")
        captions_train_cleaned_stemmed_and_lemmatized_by_spacy.append(nlp(' '.join(sentence)))
    
    captions_temp = []
    POS_captions_train = []
    for sentence in captions_train_cleaned_stemmed_and_lemmatized_by_spacy:
        captions_temp.append([token.lemma_ for token in sentence])
        POS_captions_train.append([token.pos_ for token in sentence])
    
    captions_train_cleaned_stemmed_and_lemmatized_by_spacy=captions_temp
    
    return captions_train_cleaned_stemmed_and_lemmatized_by_spacy,POS_captions_train