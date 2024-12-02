import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
from prometheus_client import start_http_server, Counter, Summary, CollectorRegistry
import time

# Custom Prometheus Registry to prevent duplicate registration issues
registry = CollectorRegistry()

# Start Prometheus metrics server
start_http_server(8000, registry=registry)

# Define Prometheus metrics
requests_count = Counter(
    'app_requests_total', 
    'Total number of requests', 
    registry=registry
)
recommendations_count = Counter(
    'recommendations_served', 
    'Number of recommendations served', 
    registry=registry
)
processing_time = Summary(
    'processing_time_seconds', 
    'Time spent processing requests', 
    registry=registry
)

st.title("Welcome to LEARN-MATCH System!")

@st.cache_data
def load_data():
    if os.path.exists("input.csv"):
        df = pd.read_csv("input.csv")
        if 'link' not in df.columns:
            df['link'] = 'https://www.skillshare.com/en/classes/Fundamentals-of-DSLR-Photography/1111783378'  # Default link
        return df
    else:
        return pd.DataFrame(columns=["title", "level", "language", "crediteligibility", "link"])

df = load_data()

def preprocess_data(df):
    df["processed_title"] = df["title"].str.lower()
    le_level = LabelEncoder()
    le_language = LabelEncoder()
    le_credit = LabelEncoder()
    if not df.empty:
        df["level_encoded"] = le_level.fit_transform(df["level"])
        df["language_encoded"] = le_language.fit_transform(df["language"])
        df["crediteligibility_encoded"] = le_credit.fit_transform(df["crediteligibility"])
    return df, le_level, le_language, le_credit

df, le_level, le_language, le_credit = preprocess_data(df)

vectorizer = TfidfVectorizer(stop_words="english")
if not df.empty:
    tfidf_matrix = vectorizer.fit_transform(df["processed_title"])
    features = np.hstack((tfidf_matrix.toarray(),
                          df[["level_encoded", "language_encoded", "crediteligibility_encoded"]].values))
    kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

topic = st.text_input("Enter the course topic:")
difficulty = st.selectbox("Select difficulty level:", ("All Levels", "Beginner", "Intermediate", "Expert"))
language = st.selectbox("Select the course language:", ("English", "Spanish", "French", "German", "Chinese"))
credit_eligibility = st.radio("Are you eligible for credits?", ("Yes", "No"))

# Track request metrics
if st.button("Get Recommendations"):
    requests_count.inc()  # Increment request count

    with processing_time.time():  # Measure processing time
        if not topic:
            st.warning("Please enter a course topic!")
        else:
            recommendations_count.inc(3)  # Example: 3 recommendations served
            
            # Process user input
            user_input = pd.DataFrame({
                "title": [topic],
                "level": [difficulty],
                "language": [language],
                "crediteligibility": [credit_eligibility],
                "link": ["https://example.com"]
            })
            user_input["processed_title"] = user_input["title"].str.lower()
            user_input["level_encoded"] = le_level.transform(user_input["level"])
            user_input["language_encoded"] = le_language.transform(user_input["language"])
            user_input["crediteligibility_encoded"] = le_credit.transform(user_input["crediteligibility"])

            # Vectorize user input
            user_tfidf = vectorizer.transform(user_input["processed_title"])
            user_features = np.hstack((user_tfidf.toarray(),
                                       user_input[["level_encoded", "language_encoded", "crediteligibility_encoded"]].values))

            # Calculate similarities
            similarities = cosine_similarity(user_features, features)
            similar_indices = similarities.argsort()[0][-10:][::-1]  # Get top 10 similar courses

            # Display recommendations
            st.subheader("Recommended Courses:")
            displayed_titles = set()
            for idx in similar_indices:
                recommended_course = df.iloc[idx]
                if recommended_course['title'] not in displayed_titles and len(displayed_titles) < 3:
                    displayed_titles.add(recommended_course['title'])
                    st.write(f"Title: {recommended_course['title']}")
                    st.write(f"Level: {recommended_course['level']}")
                    st.write(f"Language: {recommended_course['language']}")
                    st.write(f"Credit Eligible: {recommended_course['crediteligibility']}")
                    st.write(f"Link: {recommended_course['link']}")
                    st.write("---")

            # Save user input
            df = pd.concat([df, user_input], ignore_index=True)
            df.to_csv("input.csv", index=False)

st.subheader("Feedback")
feedback = st.text_area("Please provide your feedback on the recommendations:")
if st.button("Submit Feedback"):
    feedback_data = pd.DataFrame({
        "timestamp": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        "feedback": [feedback]
    })
    if os.path.exists("feedback.csv"):
        feedback_existing = pd.read_csv("feedback.csv")
        feedback_data = pd.concat([feedback_existing, feedback_data], ignore_index=True)
    feedback_data.to_csv("feedback.csv", index=False)
    st.success("Thank you for your feedback!")
