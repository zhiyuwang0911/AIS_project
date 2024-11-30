import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

st.title("Welcome to LEARN-MATCH System!")


'''
The script the deployment of the K-means course recommendation system. The use input will be added to the last row of the dataset and enumerated together. The clustering is based on the whole dataset. The recommendation is based on the top3 closest datapoint respect to the userinput datapoint use Euclidiean distance. Later the user input will be delete from the original csv file to keep the dataset clean and repeatable. To improve the trancparency of model, we use a processing bar to notify the user of our process progress.

For the performance monitering, we consider two immedate response and a delayed response. The immedate response is by monitering the total run time and the clustering runtime. The delayed monitering is by analyzing the user feedback, which is saved in csv file.
'''



# Load the dataset
@st.cache_data
def load_data():
    if os.path.exists("input.csv"):
        return pd.read_csv("input.csv")
    else:
        # Create a sample dataset if input.csv doesn't exist
        return pd.DataFrame(columns=["title", "level", "language", "crediteligibility"])

df = load_data()

# Preprocess data
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

# Vectorize titles and combine features
vectorizer = TfidfVectorizer(stop_words="english")
if not df.empty:
    tfidf_matrix = vectorizer.fit_transform(df["processed_title"])
    features = np.hstack((
        tfidf_matrix.toarray(),
        df[["level_encoded", "language_encoded", "crediteligibility_encoded"]].values
    ))
    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

# Streamlit App UI

# Input fields
topic = st.text_input("Enter the course topic:")
difficulty = st.selectbox("Select difficulty level:", ("All Levels", "Beginner", "Intermediate", "Expert"))
language = st.selectbox("Select the course language:", ("English", "Spanish", "French", "German", "Chinese"))
credit_eligibility = st.radio("Are you eligible for credits?", ("Yes", "No"))

if st.button("Get Recommendations"):
    if not topic:
        st.warning("Please enter a course topic!")
    else:
        # Process user input
        user_input = pd.DataFrame({
            "title": [topic],
            "level": [difficulty],
            "language": [language],
            "crediteligibility": [credit_eligibility]
        })
        user_input["processed_title"] = user_input["title"].str.lower()
        user_input["level_encoded"] = le_level.transform(user_input["level"])
        user_input["language_encoded"] = le_language.transform(user_input["language"])
        user_input["crediteligibility_encoded"] = le_credit.transform(user_input["crediteligibility"])

        # Vectorize user input
        user_tfidf = vectorizer.transform(user_input["processed_title"])
        user_features = np.hstack((
            user_tfidf.toarray(),
            user_input[["level_encoded", "language_encoded", "crediteligibility_encoded"]].values
        ))

        # Predict cluster for user input
        user_cluster = kmeans.predict(user_features)[0]

        # Find similar courses in the same cluster
        cluster_courses = df[df["cluster"] == user_cluster]
        if cluster_courses.empty:
            st.write("No similar courses found.")
        else:
            similarities = cosine_similarity(user_features, features[df["cluster"] == user_cluster])
            similar_indices = similarities.argsort()[0][-3:][::-1]

            # Display recommendations
            st.subheader("Recommended Courses:")
            for idx in similar_indices:
                recommended_course = cluster_courses.iloc[idx]
                st.write(f"Title: {recommended_course['title']}")
                st.write(f"Level: {recommended_course['level']}")
                st.write(f"Language: {recommended_course['language']}")
                st.write(f"Credit Eligible: {recommended_course['crediteligibility']}")
                st.write("---")

        # Save the user input to input.csv
        df = pd.concat([df, user_input], ignore_index=True)
        df.to_csv("input.csv", index=False)

# Feedback Section
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
