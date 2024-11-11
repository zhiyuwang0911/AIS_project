import streamlit as st
import pandas as pd
import joblib  # for loading the saved model
import time
import pickle
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


# Set the title of the web app
st.title("Welcome to LEARN-MATCH System !")

# Input for course topic (a text field)
topic = st.text_input("Enter the your interested topic (e.g., Python Programming):")

# Difficulty level selection (radio buttons)
difficulty = st.radio(
    "Select difficulty level:",
    ("All Levels", "Beginner", "Intermediate", "Expert")
)

# Language selection (selectbox)
language = st.selectbox(
    "Select the course language:",
    ("English", "Spanish", "French", "German", "Chinese")
)

# Credit eligibility (radio buttons)
crediteligibility = st.radio(
    "Are you eligible for credits?",
    ("Yes", "No")
)

# Submit button
if st.button("Submit"):
    # When the button is clicked, display the selected options
    st.write("You selected the following options:")
    st.write(f"Topic: {topic}")
    st.write(f"Difficulty Level: {difficulty}")
    st.write(f"Language: {language}")
    st.write(f"crediteligibility: {crediteligibility}")
# Create a dictionary with the user input
    user_input = {
        "title": [topic],
        "level": [difficulty],
        "language": [language],
        "crediteligibility": [crediteligibility]
    }

    # Convert the dictionary to a DataFrame
    df_input = pd.DataFrame(user_input)
    df_input = pd.DataFrame(user_input)

    # Check if the CSV file exists, and append data
    if os.path.isfile("input.csv"):
        df_existing = pd.read_csv("input.csv")
        df_combined = pd.concat([df_existing, df_input], ignore_index=True)
        df_combined.to_csv("input.csv", index=False)
    else:
        df_input.to_csv("input.csv", index=False)
    st.success("Data saved!")

# Step 1: Load the CSV file
if os.path.isfile("input.csv"):
    df_basic = pd.read_csv("input.csv")
    st.write("Data Loaded Successfully:")
    #st.dataframe(df_basic)
    st.title("We are customizing the course for you ......")
    progress_bar = st.progress(0)

    # Step 2: Preprocess titles (convert to lowercase)
    df_basic['processed_title'] = df_basic['title'].str.lower()
     
    # Step 3: Apply TfidfVectorizer to convert text to numerical form
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_basic['processed_title'])
    progress_bar = st.progress(10)
    
# Step 4: Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(X, X)
    progress_bar = st.progress(30)

    # Step 5: Use clustering or thresholding to group similar titles
    threshold = 0.5  # Define similarity threshold to consider titles similar
    unique_ids = []
    for i in range(len(cosine_sim)):
        found = False
        for uid in unique_ids:
            if cosine_sim[i, uid] > threshold:
                unique_ids.append(uid)
                found = True
                break
        if not found:
            unique_ids.append(i)
    
    # Step 6: Add unique_ids as a column to the dataframe
    df_basic['unique_title'] = unique_ids
    progress_bar = st.progress(40)

    # Step 7: Apply Label Encoding to categorical columns
    le = LabelEncoder()
    df_basic['level_encoded'] = le.fit_transform(df_basic['level'])
    df_basic['language_encoded'] = le.fit_transform(df_basic['language'])
    df_basic['credit_eligibility_encoded'] = le.fit_transform(df_basic['crediteligibility'])
    progress_bar = st.progress(50)
    # Display the processed DataFrame
    #st.write("Processed Data with Similar Titles Grouped and Encoded Levels:")
    #st.dataframe(df_basic[['title', 'unique_title', 'level_encoded', 'language_encoded', 'credit_eligibility_encoded']])

    # Optionally save the processed DataFrame
    #if st.button("Save Processed Data"):
    df_basic.to_csv("processed_input.csv", index=False)
    # load the save model and perform K-means clustering
    uploaded_file = "processed_input.csv"
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        #st.write("Processed data loaded:")
        #st.dataframe(df)
            
        # Step 2: Load the saved KMeans model
        #model_file = 'kmeans_optimal_basic.pkl'  # replace with your actual saved pickle file
        X = df
        kmeans = KMeans(n_clusters=3, random_state=42)  # specify the number of clusters
        kmeans.fit(X)

        # Predict the clusters for the data points
        df_basic['cluster'] = kmeans.predict(X)   
             # Step 3: Perform clustering using the loaded model
            # Exclude non-numeric columns (assuming only numerical features are relevant for clustering)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        X = df[numeric_columns]
        progress_bar = st.progress(60)
            # Predict cluster labels
        df['cluster'] = kmeans.predict(X)
        progress_bar = st.progress(80)

        input_data_point = X.iloc[[0]]  # Replace with actual user-input data point if available
        input_cluster = df['cluster'].iloc[0]
    
        # Filter only the data points within the same cluster as the input
        cluster_points = X[df['cluster'] == input_cluster]
    
        # Calculate distances to the input data point
        _, distances = pairwise_distances_argmin_min(input_data_point, cluster_points)
        progress_bar = st.progress(90)

        # Get the indices of the top 3 closest points
        closest_indices = np.argsort(distances)[:3]
        closest_courses = cluster_points.iloc[closest_indices]
        progress_bar = st.progress(95)    
        st.write("Top 3 course recomended for you:")
        st.dataframe(df.loc[closest_courses.index, ['title'] + list(numeric_columns)])
                                                                                          
