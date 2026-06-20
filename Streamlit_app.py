import streamlit as st
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

st.title("Business News Headline Classifier")

st.write("Enter a news headline and classify it into a news category.")

headline = st.text_area("News headline:")


label_names = ["World", "Sports", "Business", "Sci/Tech"]


@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_classifier = joblib.load("embedding_classifier.joblib")
    return embedding_model, embedding_classifier

@st.cache_data
def load_data():
    # Load your dataset here if needed
    return pd.read_csv("train.csv")

embedding_model, embedding_classifier = load_models()
data = load_data()

if "classified_headlines" not in st.session_state:
    st.session_state.classified_headlines = []

tab1 ,tab2 = st.tabs(["Classify Headline", "View Classified Headlines"])

with tab1:
    st.subheader("Classify a News Headline")

    headline= st.text_area("Enter a news headline to classify:")

    if st.button("Classify"):
        if headline.strip() == "":
            st.warning("Please enter a headline.")
        else:
            headline_embedding = embedding_model.encode([headline])
            prediction = embedding_classifier.predict(headline_embedding)[0]

            predicted_label = label_names[prediction]

            probabilities = embedding_classifier.predict_proba(headline_embedding)[0]
            confidence = probabilities[prediction]

            st.success(f"Predicted category: {predicted_label}")
            st.metric(f"Prediction Confidence: {confidence:.2f}")

            st.session_state.classified_headlines.append({
                "headline": headline,
                "predicted category": predicted_label,
                "confidence": confidence
            })

    st.subheader("History of Classified Headlines")
    if st.session_state.classified_headlines:
        recent_df= pd.DataFrame(st.session_state.classified_headlines)
        
        recent_category = st.selectbox(
            "Filter recent headlines by category",
            ["All"] + label_names
        )

        if recent_category != "All":
            recent_df = recent_df[
                recent_df["predicted category"]== recent_category
            ]

        recent_df["Confidence"] = recent_df["Confidence"].map(lambda x: f"{x:.2f}")

        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No Headlines Classified yet.")

    with tab2:
        st.subheader("Real headline dataset")

        st.write("Explore real headlines from your dataset.")

        category_filter = st.selectbox(
            "Filter dataset by category",
            ["All"] + sorted(data["category"].unique())
        )

        filtered_data = data.copy()

        if category_filter != "All":
            filtered_data = filtered_data[
                filtered_data["category"] == category_filter
            ]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total headlines", len(data))

        with col2:
            st.metric("Headlines shown", len(filtered_data))

        n_examples = st.slider(
            "Number of headlines to spotlight",
            min_value=5,
            max_value=50,
            value=10
        )

        spotlight = filtered_data.sample(
            min(n_examples, len(filtered_data)),
            random_state=42
        )

        st.dataframe(
            spotlight[["headline", "category"]],
            use_container_width=True
        )