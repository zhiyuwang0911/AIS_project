import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Title of the web app
st.title("Welcome to LEARN-MATCH System!")

# Input fields
topic = st.text_input("Enter your interested topic (e.g., Python Programming):")
difficulty = st.radio(
    "Select difficulty level:",
    ("All Levels", "Beginner", "Intermediate", "Expert")
)
language = st.selectbox(
    "Select the course language:",
    ("English", "Spanish", "French", "German", "Chinese")
)
credit_eligibility = st.radio(
    "Are you eligible for credits?",
    ("Yes", "No")
)

if st.button("Submit"):
    # Display user input
    st.write("You selected the following options:")
    st.write(f"Topic: {topic}")
    st.write(f"Difficulty Level: {difficulty}")
    st.write(f"Language: {language}")
    st.write(f"Credit Eligibility: {credit_eligibility}")

    # Save input to a DataFrame
    user_input = {
        "title": [topic],
        "level": [difficulty],
        "language": [language],
        "crediteligibility": [credit_eligibility],
        "link": [None]  # Placeholder for user input; links are not needed for the user's row
    }
    df_input = pd.DataFrame(user_input)

    # Check if the dataset exists
    if not os.path.isfile("input.csv"):
        st.error("Dataset not found! Please upload the 'input.csv' file with course details.")
        st.stop()

    # Load existing dataset
    df = pd.read_csv("input.csv")

    # Add user input to the dataset for processing
    df = pd.concat([df, df_input], ignore_index=True)

    # Process data
    st.write("Processing data for recommendations...")
    progress_bar = st.progress(0)

    # Text processing with TfidfVectorizer
    df['processed_title'] = df['title'].str.lower()  # Convert to lowercase for uniformity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['processed_title'])
    progress_bar.progress(30)

    # Add encoded categorical columns
    le = LabelEncoder()
    df['level_encoded'] = le.fit_transform(df['level'])
    df['language_encoded'] = le.fit_transform(df['language'])
    df['credit_eligibility_encoded'] = le.fit_transform(df['crediteligibility'])
    progress_bar.progress(60)

    # Prepare feature matrix
    tfidf_array = tfidf_matrix.toarray()
    features = np.hstack([
        tfidf_array,
        df[['level_encoded', 'language_encoded', 'credit_eligibility_encoded']].values
    ])
    progress_bar.progress(80)

    # Compute similarity
    user_features = features[-1]  # Last row is the user input
    similarity_scores = cosine_similarity([user_features], features[:-1])  # Compare against all others
    similarity_scores = similarity_scores.flatten()

    # Get top 3 recommendations (excluding the user input itself)
    top_indices = np.argsort(similarity_scores)[-3:][::-1]  # Top 3 indices sorted by similarity
    recommendations = df.iloc[top_indices][['title', 'level', 'language', 'crediteligibility', 'link']]

    progress_bar.progress(100)

    # Display recommendations
    st.write("Top 3 recommended courses based on similarity:")
    for _, row in recommendations.iterrows():
        st.markdown(f"**Title:** {row['title']}")
        st.markdown(f"- **Level:** {row['level']}")
        st.markdown(f"- **Language:** {row['language']}")
        st.markdown(f"- **Credit Eligibility:** {row['crediteligibility']}")
        st.markdown(f"- **Link to Course:**({row['link']})")
        st.markdown("---")

    # Clean up the dataset
    df = df.iloc[:-1]  # Remove user input
    df.to_csv("input.csv", index=False)

    # Feedback
    st.title("Please provide your feedback")
    if st.button("Satisfied"):
        with open("user_feedback.csv", "a") as f:
            f.write(f"{datetime.now()},Satisfied\n")
        st.write("Thank you for your feedback!")
    if st.button("Unsatisfied"):
        with open("user_feedback.csv", "a") as f:
            f.write(f"{datetime.now()},Unsatisfied\n")
        st.write("Thank you for your feedback!")
