import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('classifier.pkl', 'rb'))

st.title('Sms Spam Classifier')

sms = st.text_input('Enter your message')

#1. preprocessing the recieved sms
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

trans_sms = transform_text(sms)

#2. vectorize
vectored_sms = cv.transform([trans_sms])

#3.Predict

result = model.predict(vectored_sms)[0]

if st.button('Predict'):
    if result == 1:
        st.header('Spam')

    else:
        st.header('Not Spam')