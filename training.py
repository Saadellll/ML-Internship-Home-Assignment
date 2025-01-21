import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os

# Define the relative path for dataset loading
def load_data():
    """Load the dataset."""
    data_path = os.path.join('ML-Internship-Home-Assignment-main', 'data', 'raw', 'resume.csv')
    return pd.read_csv(data_path)

# Function to save the model and vectorizer
def save_pipeline(model, vectorizer, name):
    """Save the trained model and vectorizer."""
    model_path = os.path.join('models', f"{name}_model.joblib")
    vectorizer_path = os.path.join('models', f"{name}_vectorizer.joblib")
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)

def render_training():
    """Render the Training section."""
    st.header("Pipeline Training")
    st.info(
        "Before you proceed to training your pipeline, make sure you "
        "have checked your training pipeline code and that it is set properly."
    )

    name = st.text_input("Pipeline name", placeholder="Logistic Regression")
    serialize = st.checkbox("Save pipeline")
    train = st.button("Train pipeline")

    if train:
        with st.spinner("Training pipeline, please wait..."):
            try:
                # Load the data using the function
                data = load_data()

                # Feature extraction using TF-IDF Vectorizer
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                X = vectorizer.fit_transform(data['Resume Text'])
                y = data['Label']  # 'Label' is directly taken from the dataset

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Logistic Regression Model
                model = LogisticRegression(max_iter=1000)

                # Hyperparameter tuning with GridSearchCV
                param_grid = {
                    'C': [0.1, 1, 10],  # Regularization parameter
                    'solver': ['liblinear', 'saga'],  # Solvers to try
                }
                grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Best model
                best_model = grid_search.best_estimator_

                # Model evaluation
                y_pred = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Show metrics
                col1, col2 = st.columns(2)
                col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
                col2.metric(label="F1 score", value=str(round(f1, 4)))

                # Display confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data['Label'].unique(), yticklabels=data['Label'].unique())
                st.pyplot(fig)

                # Option to save the trained pipeline
                if serialize:
                    # Save the model and vectorizer
                    save_pipeline(best_model, vectorizer, name)

            except Exception as e:
                st.error("Failed to train the pipeline!")
                st.exception(e)
