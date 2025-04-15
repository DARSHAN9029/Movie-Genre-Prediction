import streamlit as st
import pickle

#load model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('genre_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Text cleaning function (use the same one you used during training
def cleaned_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Movie Genre Predictor")                        #Streamlit UI
st.markdown("Enter a **movie title and description**, and I'll predict the genre!")

title = st.text_input("Movie Title")                            #Input fields
description = st.text_area("Movie Description")

if st.button("Predict Genre"):
    combined_text = f"{title} {description}"
    cleaned = cleaned_text(combined_text)
    transformed = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(transformed)
    st.success(f"Predicted Genre: {prediction[0]}")