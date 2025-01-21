import streamlit as st
import requests
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Initialize the database
Base = declarative_base()

class InferenceResult(Base):
    """Define the database model for storing inference results."""
    __tablename__ = 'inference_results'
    id = Column(Integer, primary_key=True)
    resume_sample = Column(String, nullable=False)
    predicted_label = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create SQLite database (if it doesn't exist)
DATABASE_URL = "sqlite:///inference_results.db"
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(bind=engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

# Assuming LABELS_MAP is already available from the data_ml_assignment.constants
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH

def save_inference_result(resume_sample, predicted_label):
    """Save the prediction results to the SQLite database."""
    result = InferenceResult(resume_sample=resume_sample, predicted_label=predicted_label)
    session.add(result)
    session.commit()

def render_inference():
    """Render the Inference section."""
    st.header("Resume Inference")
    st.info(
        "This section simplifies the inference process. "
        "Choose a test resume and observe the label that your trained pipeline will predict."
    )

    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button("Run Inference")

    if infer:
        with st.spinner("Running inference..."):
            try:
                # Prepare the sample filename
                sample_file = "_".join(sample.upper().split()) + ".txt"
                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                # Perform inference via API request
                result = requests.post(
                    "http://localhost:9000/api/inference", json={"text": sample_text}
                )
                st.success("Inference completed!")

                # Extract the predicted label from the result
                label = LABELS_MAP.get(int(float(result.text)))  # Assuming result is numeric and in float form
                st.metric(label="Predicted Label", value=label)

                # Save the inference result into SQLite
                save_inference_result(sample, label)

                # Show the contents of the inference results table
                st.subheader("Inference Results Table")
                results = session.query(InferenceResult).all()
                table_data = [(res.resume_sample, res.predicted_label, res.timestamp) for res in results]
                st.write(table_data)

            except Exception as e:
                st.error("Failed to call Inference API!")
                st.exception(e)
