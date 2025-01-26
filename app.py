import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

import streamlit as st
import pickle 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    alpha_numeric_char = [w for w in text if w.isalnum()]
    after_removing_stop_words = [w for w in alpha_numeric_char 
                                 if w not in stopwords.words('english')]
    after_lemmatization = [lm.lemmatize(w) for w in after_removing_stop_words]

    return after_lemmatization

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Detector')
st.write('This is a simple SMS Spam Detector using NLP')
st.write('By Bhumika')

input = st.text_input("Please enter your SMS")
input_sms = input.strip() 
input_sms = input_sms.encode("ascii", "ignore").decode()


if st.button('Predict'):
    transformed_sms = " ".join(transform_text(input_sms))
    vectorized_sms = cv.transform([transformed_sms])
    #st.write(transform_sms)
    #vectorized_sms = cv.transform([transform_sms])
    #st.write(vectorized_sms)
    prediction = model.predict(vectorized_sms)[0]
    print(prediction)
    #st.write(prediction)

    if prediction == 0:
        st.write('This SMS is SPAM')
    else:
        st.write('This SMS is not SPAM')

    



