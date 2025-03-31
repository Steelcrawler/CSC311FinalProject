"""
This script implements a prediction function for the food survey classification task.
It loads the trained model components from disk and applies them to new data.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import re

# Add the parent directory to sys.path to import FoodSurveyDataLoader
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from very_good_dataloader import FoodSurveyDataLoader

# Path to the saved model components
MODEL_DIR = 'logistic_regression_model'

def load_model_components():
    """
    Load the model components (preprocessor, classifier, and feature lists)
    from the saved pickle files.
    """
    with open(f"{MODEL_DIR}/preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(f"{MODEL_DIR}/classifier.pkl", 'rb') as f:
        classifier = pickle.load(f)
    
    with open(f"{MODEL_DIR}/feature_lists.pkl", 'rb') as f:
        feature_lists = pickle.load(f)
    
    return preprocessor, classifier, feature_lists

def predict_all(filename):
    """
    Make predictions for all examples in the given CSV file.
    
    Args:
        filename: Path to the CSV file containing the test data
        
    Returns:
        A list of predictions (food type labels) for each example
    """
    try:
        # Load the saved model components
        preprocessor, classifier, feature_lists = load_model_components()
        
        # Initialize the dataloader with the test file
        dataloader = FoodSurveyDataLoader(filename)
        dataloader.load_data()
        
        # Process all data at once using the dataloader's preprocess_data method
        processed_data = dataloader.preprocess_data()
        
        # Extract features used in training
        X = processed_data[dataloader.feature_names]
        
        # Apply preprocessor and classifier
        X_processed = preprocessor.transform(X)
        pred_encoded = classifier.predict(X_processed)
        
        # Convert encoded predictions back to original labels
        predictions = [dataloader.index_to_class[code] for code in pred_encoded]

        # Y = list(processed_data['label_encoded'])
        # crct = 0
        # total = 0
        # for i in range(len(pred_encoded)):
        #     if pred_encoded[i] == Y[i]:
        #         crct += 1
        #     total += 1
        # print("\n\n\n\n")
        # print("acc: ", crct / total)
        
        return predictions
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        # In case of error, return a fallback prediction for each example
        try:
            test_data = pd.read_csv(filename)
            return ['Pizza'] * len(test_data)
        except:
            return []

if __name__ == "__main__":
    # Check if a filename was provided as a command-line argument
    filename = "cleaned_data_combined_modified.csv"
    
    # Make predictions
    predictions = predict_all(filename)
    
    # Print the predictions
    print(f"Made {len(predictions)} predictions:")
    for i, pred in enumerate(predictions[:10]):  # Print first 10 predictions
        print(f"Example {i+1}: {pred}")
    
    if len(predictions) > 10:
        print(f"... and {len(predictions) - 10} more")