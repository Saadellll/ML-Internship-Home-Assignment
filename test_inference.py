import pytest
from unittest import mock
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from data_ml_assignment.inference import render_inference, save_inference_result
from data_ml_assignment.constants import LABELS_MAP, SAMPLES_PATH
from data_ml_assignment.models import InferenceResult, Base
import requests
import io

# Mocking the file path for test purposes
SAMPLES_PATH = "mock_samples"  # Define a mock path for testing

# Mock for the Inference API response
@pytest.fixture
def mock_inference_api():
    with mock.patch('requests.post') as mock_post:
        # Mock a successful inference response
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "1"  # Mocking the predicted label (assuming '1' corresponds to "Dot Net Developer")
        yield mock_post


# Setting up an in-memory SQLite database for testing
@pytest.fixture(scope="module")
def test_db():
    """Setup an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")  # In-memory database
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


# Test saving inference result to the database
def test_save_inference_result(test_db):
    """Test saving inference results to the database."""
    sample = "JAVA_DEVELOPER"
    predicted_label = "Java Developer"

    save_inference_result(test_db, sample, predicted_label)  # Save using the mock database session

    # Query the database and verify the result
    result = test_db.query(InferenceResult).filter_by(resume_sample=sample).first()
    assert result is not None
    assert result.predicted_label == predicted_label
    assert isinstance(result.timestamp, datetime)  # Check if timestamp is correctly added


# Test the render_inference logic with the mock API response
def test_render_inference(mock_inference_api, test_db):
    """Test the full render_inference function logic."""
    # Mocking the Streamlit interface
    with mock.patch('streamlit.selectbox') as mock_selectbox, \
         mock.patch('streamlit.spinner') as mock_spinner, \
         mock.patch('streamlit.success') as mock_success, \
         mock.patch('streamlit.metric') as mock_metric:

        # Mock resume sample selection
        mock_selectbox.return_value = "JAVA_DEVELOPER"

        # Run the inference logic
        render_inference()

        # Check if the mock API was called
        mock_inference_api.assert_called_once()

        # Verify the expected interactions with Streamlit
        mock_spinner.assert_called_once()  # Ensure spinner was used during the inference
        mock_success.assert_called_once_with("Inference completed!")
        mock_metric.assert_called_once_with(label="Predicted Label", value="Java Developer")

        # Check if the result was saved into the database
        result = test_db.query(InferenceResult).filter_by(resume_sample="JAVA_DEVELOPER").first()
        assert result is not None
        assert result.predicted_label == "Java Developer"


# Test database integrity - no duplicate records should be allowed
def test_save_inference_result_duplicate(test_db):
    """Test that saving duplicate results raises an integrity error."""
    sample = "JAVA_DEVELOPER"
    predicted_label = "Java Developer"
    
    # Save the first result
    save_inference_result(test_db, sample, predicted_label)
    
    # Attempt to save a duplicate record, which should raise an IntegrityError
    with pytest.raises(IntegrityError):
        save_inference_result(test_db, sample, predicted_label)
