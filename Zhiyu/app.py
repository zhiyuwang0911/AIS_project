import streamlit as st
from datetime import datetime
import pandas as pd
import joblib  # for loading the saved model
import numpy as np
import time
import pickle
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
'''
The script the deployment of the K-means course recommendation system. The use input will be added to the last row of the dataset and enumerated together. The clustering is based on the whole dataset. The recommendation is based on the top3 closest datapoint respect to the userinput datapoint use Euclidiean distance. Later the user input will be delete from the original csv file to keep the dataset clean and repeatable. To improve the trancparency of model, we use a processing bar to notify the user of our process progress.

For the performance monitering, we consider two immedate response and a delayed response. The immedate response is by monitering the total run time and the clustering runtime. The delayed monitering is by analyzing the user feedback, which is saved in csv file.
'''

# immedate performane monitering
def monitor_run_time(df):
    # Track the total runtime of the script
    start_script_time = time.time()

    # Track the clustering time
    start_clustering_time = time.time()

    # Simulate clustering with KMeans
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_columns]
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Calculate clustering time
    clustering_time = time.time() - start_clustering_time

    # Calculate total runtime of the script
    total_runtime = time.time() - start_script_time
    
    return clustering_time, total_runtime




# User interface

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
        X_text = vectorizer.fit_transform(df_basic['processed_title'])
        X = X_text

        #print(X)
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
            columns_to_drop = ['title','level','language','crediteligibility','processed_title','credit_eligibility_encoded']
            df = df.drop(columns=columns_to_drop)
            df.dropna(inplace=True)
            df.rename(columns={'level_encoded': 'level', 'language_encoded': 'language','crediteligibility_encodede':'crediteligibility','unique_title':'title'}, inplace=True)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
           #st.write("Processed data loaded:")
           #st.dataframe(df)
            print(df.columns)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
        # Step 2: Load the saved KMeans model
        #model_file = 'kmeans_optimal_basic.pkl'  # replace with your actual saved pickle file
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            X = df[numeric_columns]
            kmeans = KMeans(n_clusters=2, random_state=42)  # specify the number of clusters
            kmeans.fit(X)

        # Predict the clusters for the data points
             # Step 3: Perform clustering using the loaded model
            # Exclude non-numeric columns (assuming only numerical features are relevant for clustering)
            #numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            progress_bar = st.progress(60)
            # Predict cluster labels
            df['cluster'] = kmeans.predict(X)
            df = df.loc[:,~df.columns.duplicated()]      
            input_data_point = X.iloc[[0]]  # Replace with actual user-input data point if available
            input_cluster = df['cluster'].iloc[0]
            progress_bar = st.progress(80)
            print(df.columns)    
            # Filter only the data points within the same cluster as the input
            cluster_points = X[df['cluster'] == input_cluster]
    
            # Calculate distances to the input data point
            _, distances = pairwise_distances_argmin_min(input_data_point, cluster_points)
            progress_bar = st.progress(90)

            # Get the indices of the top 3 closest points
            #print(df)
            closest_indices = np.argsort(distances)[:3]  # Sort distances and pick the closest ones
            closest_courses = cluster_points.iloc[closest_indices]

           # Step 4: Display the titles of the closest points
            closest_titles = df.loc[closest_courses.index, 'title']

           # Show the results in Streamlit
            st.write("Top 3 recommended courses based on similarity:")
            st.dataframe(closest_titles)         

           #delete the input datapoint from dataset to make the result repetable 
            def delete_last_row(file_path):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
    
                # Drop the last row (using .iloc to exclude the last row)
                df = df.iloc[:-1]
    
                # Save the updated DataFrame back to the same CSV file
                df.to_csv(file_path, index=False)

           # Delete the last row from input.csv only
            delete_last_row('input.csv')

           # collect the user feedback and time

           

           # Function to save user feedback to a CSV file
            def save_feedback(feedback):
           # Get the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
            # Create a DataFrame to store the feedback and timestamp
                feedback_data = pd.DataFrame({
                    'timestamp': [timestamp],
                    'feedback': [feedback]
                })
    
            # Save feedback to a CSV file for the delayed performance monitering
                feedback_data.to_csv('user_feedback.csv', mode='a', header=False, index=False)
                st.write(f"Feedback recorded: {feedback} at {timestamp}")

            # Streamlit app interface
            st.title("Please provide your feedback")

             # Button for satisfied feedback
            if st.button("Satisfied"):
                save_feedback("Satisfied")

            # Button for unsatisfied feedback
            if st.button("Unsatisfied"):
                save_feedback("Unsatisfied")

            st.write("Your feedback will help us improve!")


                                                                                                     
