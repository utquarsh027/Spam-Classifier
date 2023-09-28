import streamlit as st
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pickle
import nltk

ps=PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
tfidf = pickle.load(open('vector.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
st.header('Email/SMS Classifer')

input_sms=st.text_area(label=' ',placeholder="Enter your text here")

if st.button("Predict"):
    transformed_text=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_text])
    result=model.predict(vector_input)[0]
    if result==1:
        st.write('Spam')
    else:
        st.write('Not Spam')    
import sys
print(sys.executable)        
