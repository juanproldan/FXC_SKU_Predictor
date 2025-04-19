import requests
import json
import time
from datetime import datetime

# Base URL for the API
BASE_URL = "http://127.0.0.1:5000"

def test_prediction():
    """Test the prediction endpoint."""
    print("Testing prediction endpoint...")
    
    # Test data
    data = {
        "maker": "RENAULT",
        "series": "LOGAN",
        "model_year": "2015",
        "description": "amortiguador delantero renault logan"
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction successful!")
        print(f"Predicted SKU: {result.get('sku')}")
        print(f"Confidence: {result.get('confidence'):.3f}")
        print(f"Model used: {result.get('model_used')}")
        print(f"Top SKUs: {len(result.get('top_skus', []))} alternatives")
        return result
    else:
        print(f"Prediction failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_feedback(prediction_result):
    """Test the feedback endpoint."""
    if not prediction_result:
        print("Skipping feedback test because prediction failed.")
        return
    
    print("\nTesting feedback endpoint...")
    
    # Test data for correct prediction
    correct_feedback = {
        "description": "amortiguador delantero renault logan",
        "maker": "RENAULT",
        "series": "LOGAN",
        "model_year": "2015",
        "predicted_sku": prediction_result.get('sku'),
        "is_correct": True,
        "confidence": prediction_result.get('confidence'),
        "timestamp": datetime.now().isoformat()
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/api/feedback", json=correct_feedback)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print(f"Feedback submission successful!")
        print(f"Response: {result}")
    else:
        print(f"Feedback submission failed with status code {response.status_code}")
        print(f"Response: {response.text}")
    
    # Test data for incorrect prediction
    incorrect_feedback = {
        "description": "emblema renault para tapa baul",
        "maker": "RENAULT",
        "series": "LOGAN",
        "model_year": "2015",
        "predicted_sku": "7711375979",
        "is_correct": False,
        "correct_sku": "8200798507",
        "confidence": 0.516,
        "timestamp": datetime.now().isoformat()
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/api/feedback", json=incorrect_feedback)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print(f"Incorrect feedback submission successful!")
        print(f"Response: {result}")
    else:
        print(f"Incorrect feedback submission failed with status code {response.status_code}")
        print(f"Response: {response.text}")

def test_feedback_stats():
    """Test the feedback stats endpoint."""
    print("\nTesting feedback stats endpoint...")
    
    # Make the request
    response = requests.get(f"{BASE_URL}/api/feedback/stats")
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print(f"Feedback stats retrieval successful!")
        print(f"Total feedback: {result.get('total_count')}")
        print(f"Correct predictions: {result.get('correct_count')}")
        print(f"Incorrect predictions: {result.get('incorrect_count')}")
        print(f"Accuracy rate: {result.get('accuracy_rate'):.2f}")
        print(f"Recent feedback: {len(result.get('recent_feedback', []))} items")
    else:
        print(f"Feedback stats retrieval failed with status code {response.status_code}")
        print(f"Response: {response.text}")

def main():
    """Run all tests."""
    print("Starting tests...")
    
    # Test prediction
    prediction_result = test_prediction()
    
    # Wait a bit to ensure the prediction is processed
    time.sleep(1)
    
    # Test feedback
    test_feedback(prediction_result)
    
    # Wait a bit to ensure the feedback is processed
    time.sleep(1)
    
    # Test feedback stats
    test_feedback_stats()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
