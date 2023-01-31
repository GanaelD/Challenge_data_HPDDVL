#Challenge Data 
#NLP challenge

import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import spacy 
from spacy.lang.fr.examples import sentences

#On importe les fichiers csv et on en fait des tableaux numpy

#Données pour entrainement --> Réponse 2 à 596
dataXTrain = pd.read_csv("X_train.csv", delimiter= ';')
dfDataXTrain = pd.DataFrame(data=dataXTrain)
X_train = np.array(dataXTrain)

#Données pour entrainement --> Catégorie des réponse de XTrain
dataYTrain = pd.read_csv("y_train.csv", delimiter= ';')
Y_train = np.array(dataYTrain)

#Réponse 599 à 802 auxquelles on doit prédire la catégorie
dataXTest = pd.read_csv("X_test.csv", delimiter= ';')
X_test = np.array(dataXTest)

#Données supplémentaire non labelisé pour entrainement supplémentaire si besoin
dataMoreData = pd.read_csv("nonlabeled_data.csv", delimiter= ';')
MoreData = np.array(dataMoreData)


phrase = "Ceci est une phrase que l'on va tokeniser. Ensuite on va enlever les mots inutiles."

#On tokenise la phrase en mots
words = nltk.word_tokenize(phrase)
#print(words)

#Ensuite, on va enlever les mots non significatif, les stopwords

#On enleve la ponctuation
#On a une liste de base qu'on essaye d'enrichir pour être le plus efficace possible
#print(stopwords.words("french"))
stopwords_french = stopwords.words("french")
stopwords_french.extend(string.punctuation)
other_stopwords = ["alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mien", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "ou", "où", "par", "parce", "pas", "peut", "peu", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez"	, "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être", "l'on"]
stopwords_french.extend(other_stopwords)

#Ensuite, on enlève les mots de la liste de phrase qui sont dans notre liste de stopwords
words_cleaned = [word for word in words if word not in stopwords_french]


#On applique ensuite un filtre de lemmatization, pour que les mots ayant le meme sens soit assimilé à la meme racine
#Cependant, on doit apposer un tag à chaque mot pour savoir à quel catégorie il appartient

#Noun (Singular)	NN
#Noun (Plural)	NNS
#Verb	VB
#Determiner	DT
#Adjective	JJ
#Adverb	RB

#On peut utiliser nltk pour lemmatizer
#lemmatizer = WordNetLemmatizer()
#taggedNLTK = nltk.pos_tag(words_cleaned)

#On va utiliser spacy qui est meilleur pour lemmatiser en francais
NLPSpacy = spacy.load('fr_core_news_md')
#python -m spacy download fr_core_news_md
#'fr_core_news_md'   autre version

taggedSpacy = []
phraseClean = ""

for token in words_cleaned:
    #on join les mots nettoyé en une phrase qu'on rentre ensuite dans NLPSpacy
    phraseClean += token + " "
phraseClean = NLPSpacy(phraseClean) 

#Ensuite, on va calculer le lemma de chaque mot et l'ajouter dans lemmatizedWordSpacy
lemmatizedWordSpacy = []
for token in phraseClean:
    taggedSpacy.append((token.text, token.pos_))
    lemmatizedWordSpacy.append(token.lemma_)

print(taggedSpacy)
print(lemmatizedWordSpacy)


#On fait maintenant ça pour les dataset qu'on veut utiliser

def return_token(sentence):
    doc=nltk.word_tokenize(sentence, language='french')
    return [X for X in doc]

def treat_data(dataFrame):
    
    #tokenize phrase
    captions_tokenized = [return_token(line) for line in dataFrame['Caption']]
    
    #on enlève les stopwords de chaque phrase
    captions_cleaned = []
    for message in captions_tokenized:
        captions_cleaned.append([word.lower() for word in message if word not in stopwords_french])
    
    #On joint les token pour en faire des phrases
    captions_lemmatized = []

    for sentence in captions_cleaned:
        captions_lemmatized.append(NLPSpacy(' '.join(sentence)))
    
    #On lemmatize chaque phrase
    captions_temp = []
    for sentence in captions_lemmatized:
        sentence_lemmas = []
        for token in sentence:
            if token.lemma_ not in stopwords_french:
                sentence_lemmas.append(token.lemma_)
        captions_temp.append(sentence_lemmas)
    
    captions_lemmatized=captions_temp
    
    lemmas = pd.Series(captions_lemmatized, name='lemmas', dtype='object')
    
    return lemmas


DataXTrain_cleaned = treat_data(dfDataXTrain)
print(DataXTrain_cleaned)