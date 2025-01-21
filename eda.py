import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Function to load dataset with relative path
def load_data():
    """Load the dataset."""
    # Use a relative path to the dataset
    data_path = os.path.join('ML-Internship-Home-Assignment-main', 'data', 'raw', 'resume.csv')
    return pd.read_csv(data_path)

def display_label_distribution(data):
    """Display label distribution chart."""
    label_counts = data['Label'].value_counts()
    st.bar_chart(label_counts)

def display_wordcloud(data):
    """Display word cloud of resume text."""
    all_text = " ".join(data['Resume Text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    st.image(wordcloud.to_image())

def render_eda():
    """Render the Exploratory Data Analysis section."""
    st.header("Exploratory Data Analysis")
    st.info("In this section, you are invited to explore and create insightful graphs about the resume dataset.")

    # Load the data
    data = load_data()

    # Display Data Preview
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # Display Statistical Summary
    st.subheader("Statistical Summary")
    st.write(data.describe())

    # Show Label Distribution
    st.subheader("Label Distribution")
    display_label_distribution(data)

    # Option to show a WordCloud of resume text
    show_wordcloud = st.checkbox("Show Word Cloud of Resume Text")
    if show_wordcloud:
        st.subheader("Word Cloud")
        display_wordcloud(data)

    # Add more visualizations (Histograms, Boxplots, etc.)
    st.subheader("Resume Length Distribution")
    data['Resume Length'] = data['Resume Text'].apply(lambda x: len(str(x).split()))

    # Plot histogram using matplotlib and display it with st.pyplot
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Resume Length'], bins=20, kde=True, color='blue')
    plt.title('Resume Length Distribution')
    plt.xlabel('Resume Length (in words)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
