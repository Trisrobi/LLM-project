import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

st.title("Business News Headline Classifier")

st.write("Enter a news headline and classify it into a news category.")

headline = st.text_area("News headline:")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_classifier = joblib.load("embedding_classifier.joblib")

label_names = ["World", "Sports", "Business", "Sci/Tech"]

if st.button("Classify"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        headline_embedding = embedding_model.encode([headline])
        prediction = embedding_classifier.predict(headline_embedding)[0]

        predicted_label = label_names[prediction]

        st.success(f"Predicted category: {predicted_label}")