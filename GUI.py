# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: siddhardhan
"""
import json
import re

import nltk
import numpy as np
import pickle

import pandas as pd
import streamlit as st
import tokenizer as tokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow import keras
from textblob import TextBlob, Word
from nltk.corpus import stopwords
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# loading the saved model
loaded_model = keras.models.load_model('./model-3-64-thebest.h5')
# creating a function for Prediction

def lemma(text):  # Lemmatization of cleaned body
    sent = TextBlob(text)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    seperator = ' '
    lemma = seperator.join(lemmatized_list)
    return lemma

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)  # line breaks
    # text = re.sub(r"\'\xa0", " ", text) # xa0 Unicode representing spaces
    # text = re.sub('\s+', ' ', text) # one or more whitespace characters
    text = text.strip(' ')  # spaces
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # lemmatize and remove stopwords
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    text = ' '.join(no_stopword_text)

    return text

def top_predictions(df, n):
    n = int(n)
    cols = df.columns[:].tolist()
    print(cols)
    a = df[cols].to_numpy().argsort()[:, :-n-1:-1]
    print(a)
    c = np.array(cols)[a]
    print(c)
    d = df[cols].to_numpy()[np.arange(a.shape[0])[:, None], a]

    df1 = pd.DataFrame(c).rename(columns=lambda x : f'max_{x+1}_col')
    cols = df1.columns[:].tolist()
    predicted_genres = ""
    for col in cols:
        predicted_genres = predicted_genres + df1[col] + " "
    return predicted_genres


def genre_prediction(text, n_top, file):

    if(not file):
        dataframe = pd.DataFrame([text], columns=['synopsis'])
        dataframe['synopsis'].apply(lambda x: preprocess_text(x))
        dataframe['synopsis'].apply(lambda x: lemma(x))
        text = dataframe['synopsis']
        st.write(text)
        len_text = 0
    else :
        len_text = len(text)
    #
    # train = pd.read_csv('C:/Users/tegar/Downloads/NLP--film-genres-from-synopsis-main/NLP--film-genres-from-synopsis-main/data/train.csv')
    # text['clean_plot'] = text['synopsis'].apply(lambda x: preprocess_text(x))
    # text['lemma'] = text['clean_plot'].apply(lambda x: lemma(x))
    # X = text['lemma']

    with open('./tokenizer.json') as f:
        data = json.load(f)
        tokenizer = keras.preprocessing.text.tokenizer_from_json(data)


    # max_features = 5000
    # tokenizer.fit_on_texts(list(text))
    tokenized_text = tokenizer.texts_to_sequences(text)
    X_te = pad_sequences(tokenized_text, maxlen=200)
    predicted = loaded_model.predict(X_te, batch_size=64, verbose=1)
    predicted = predicted[:len_text+1]
    st.write(predicted)
    df_probs_all = pd.DataFrame(predicted, columns=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'])
    pred_gen = top_predictions(df_probs_all,n_top)
    submission = pd.DataFrame(data={'predicted_genres': pred_gen})
    file = convert_df(submission)
    st.table(submission)
    st.download_button(
        label="Download data as CSV",
        data=file,
        file_name='large_df.csv',
        mime='text/csv',
    )


def uploud_button():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        dataframe['synopsis'].apply(lambda x: preprocess_text(x))
        dataframe['synopsis'].apply(lambda x: lemma(x))
        text = dataframe['synopsis']
        st.write(text)
        genre_prediction(text, 5, True)

def main():
    # giving a title
    st.title('Genre Prediction')


    # getting the input data from the user

    
    text_input = st.text_input('Text')
    n_top_input = st.number_input('N-Top', value=5)


    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button(f'Predict Genre'):
        genre_prediction(text_input,n_top_input,False)

    uploud_button()


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
  
    
  