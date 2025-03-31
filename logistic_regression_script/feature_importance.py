import os
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
from collections import Counter
from predict import FoodSurveyDataLoader, load_model_components, FeatureColumnMismatchHandler

def empty_column(df, column):
    """
    Creates a copy of the dataframe with the specified column emptied
    """
    df_copy = df.copy()
    df_copy[column] = np.nan if column != 'Q3: In what setting would you expect this food to be served? Please check all that apply' else ''
    return df_copy

def test_feature_importance(filename):
    """
    Tests the model's accuracy when each input column is emptied one at a time
    
    Args:
        filename: Path to the CSV file containing the test data with true labels
    """
    try:
        # Load the original data with true labels
        df = pd.read_csv(filename)
        if 'Label' not in df.columns:
            print("Error: The input CSV must contain a 'Label' column with true labels")
            return
        
        # Get the original column names from the data
        original_columns = [
            'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
            'Q2: How many ingredients would you expect this food item to contain?',
            'Q3: In what setting would you expect this food to be served? Please check all that apply',
            'Q4: How much would you expect to pay for one serving of this food item?',
            'Q5: What movie do you think of when thinking of this food item?',
            'Q6: What drink would you pair with this food item?',
            'Q7: When you think about this food item, who does it remind you of?',
            'Q8: How much hot sauce would you add to this food item?'
        ]
        
        # Load model components
        preprocessor, classifier, feature_lists, expected_columns = load_model_components()
        
        # Dictionary to store accuracy results
        accuracy_results = {}
        
        # First, calculate baseline accuracy with all columns intact
        dataloader = FoodSurveyDataLoader(filename)
        dataloader.load_data()
        processed_data = dataloader.preprocess_data()
        
        X = processed_data[dataloader.feature_names]
        
        # Handle feature mismatch
        feature_handler = FeatureColumnMismatchHandler(expected_columns)
        X_aligned = feature_handler.transform(X)
        
        X_processed = preprocessor.transform(X_aligned)
        pred_encoded = classifier.predict(X_processed)
        
        # Map predictions back to labels
        index_to_class = dataloader.index_to_class
        predictions = [index_to_class.get(code, 'Unknown') for code in pred_encoded]
        
        # Calculate baseline accuracy
        correct = sum(1 for pred, true in zip(predictions, df['Label']) if pred == true)
        baseline_accuracy = correct / len(df)
        print(f"\nBaseline accuracy (all columns): {baseline_accuracy:.4f}")
        
        # Test each column's importance by emptying it
        for col in original_columns:
            print(f"\nTesting importance of column: {col}")
            
            # Create a temporary CSV with the current column emptied
            temp_df = empty_column(df, col)
            temp_filename = "temp_column_empty.csv"
            temp_df.to_csv(temp_filename, index=False)
            
            # Process the data with the empty column
            temp_dataloader = FoodSurveyDataLoader(temp_filename)
            temp_dataloader.load_data()
            temp_processed = temp_dataloader.preprocess_data()
            
            X_temp = temp_processed[temp_dataloader.feature_names]
            
            # Handle feature mismatch
            X_temp_aligned = feature_handler.transform(X_temp)
            
            # Make predictions
            X_temp_processed = preprocessor.transform(X_temp_aligned)
            temp_pred_encoded = classifier.predict(X_temp_processed)
            
            # Map predictions back to labels
            temp_predictions = [index_to_class.get(code, 'Unknown') for code in temp_pred_encoded]
            
            # Calculate accuracy
            correct = sum(1 for pred, true in zip(temp_predictions, df['Label']) if pred == true)
            accuracy = correct / len(df)
            
            # Calculate accuracy drop
            accuracy_drop = baseline_accuracy - accuracy
            accuracy_results[col] = (accuracy, accuracy_drop)
            
            print(f"Accuracy with empty {col}: {accuracy:.4f}")
            print(f"Accuracy drop: {accuracy_drop:.4f}")
            
            # Show the confusion matrix
            confusion = {}
            for true_label in set(df['Label']):
                confusion[true_label] = Counter()
                
            for pred, true in zip(temp_predictions, df['Label']):
                confusion[true][pred] += 1
                
            print("\nConfusion matrix:")
            print("True Label | Predicted Labels")
            print("-" * 50)
            for true_label, counts in confusion.items():
                print(f"{true_label:10} | {dict(counts)}")
            
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        # Print summary of results
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE SUMMARY")
        print("=" * 60)
        print("Column | Accuracy | Accuracy Drop")
        print("-" * 60)
        
        # Sort by accuracy drop (descending)
        sorted_results = sorted(accuracy_results.items(), key=lambda x: x[1][1], reverse=True)
        
        for col, (accuracy, drop) in sorted_results:
            short_col = col.split(':')[0] if ':' in col else col
            print(f"{short_col:10} | {accuracy:.4f} | {drop:.4f}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    filename = "cleaned_data_combined_modified.csv"
    
    test_feature_importance(filename)