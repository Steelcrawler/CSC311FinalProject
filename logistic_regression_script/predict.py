import os
import re
import pickle
import numpy as np
import pandas as pd

def extract_number(value):
    """Extract numerical values from strings."""
    if pd.isna(value) or not isinstance(value, str):
        return value

    matches = re.findall(r'(\d+(\.\d+)?)', value)
    if matches and len(matches) > 0:
        return float(matches[0][0])
    return np.nan

def preprocess_data(df, feature_lists):
    """
    Preprocess input data using saved feature lists and preprocessing logic.
    
    Parameters:
    - df (pandas.DataFrame): Input dataframe to preprocess
    - feature_lists (dict): Dictionary containing saved feature lists
    
    Returns:
    - pandas.DataFrame: Preprocessed dataframe
    """
    # Rename columns to match original preprocessing
    columns_mapping = {
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'complexity',
        'Q2: How many ingredients would you expect this food item to contain?': 'ingredients',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'settings',
        'Q4: How much would you expect to pay for one serving of this food item?': 'price',
        'Q5: What movie do you think of when thinking of this food item?': 'movie',
        'Q6: What drink would you pair with this food item?': 'drink',
        'Q7: When you think about this food item, who does it remind you of?': 'reminds_of',
        'Q8: How much hot sauce would you add to this food item?': 'hot_sauce'
    }

    for old_col, new_col in columns_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    processed_df = pd.DataFrame(index=df.index)

    # Complexity
    if 'complexity' in df.columns:
        processed_df['complexity'] = df['complexity'].copy()

    # Ingredients
    if 'ingredients' in df.columns:
        processed_df['ingredients_num'] = df['ingredients'].apply(extract_number)

    # Price
    if 'price' in df.columns:
        processed_df['price_num'] = df['price'].apply(extract_number)

    # Hot Sauce Level
    if 'hot_sauce' in df.columns:
        hot_sauce_map = {
            'None': 0,
            'A little (mild)': 1,
            'A moderate amount (medium)': 2,
            'A lot (hot)': 3,
            'I will have some of this food item with my hot sauce': 4
        }
        processed_df['hot_sauce_level'] = df['hot_sauce'].map(hot_sauce_map).fillna(-1)

    # Process settings (multiple choice)
    if 'settings' in df.columns:
        common_settings = [
            'Week day lunch', 'Week day dinner', 'Weekend lunch',
            'Weekend dinner', 'At a party', 'Late night snack'
        ]

        for setting in common_settings:
            col_name = f'setting_{setting.lower().replace(" ", "_")}'
            processed_df[col_name] = df['settings'].apply(
                lambda x: 1 if isinstance(x, str) and setting in x else 0
            )

    if 'movie' in df.columns:
        df['movie_cleaned'] = df['movie'].fillna('').str.lower()
        
        movie_words = df['movie_cleaned'].str.split(expand=True).stack()
        top_movie_words = movie_words.value_counts().head(20).index

        for word in top_movie_words:
            col_name = f'movie_word_{word}'
            processed_df[col_name] = df['movie_cleaned'].apply(
                lambda x: 1 if word in x else 0
            )

    if 'drink' in df.columns:
        processed_df['drink_category'] = 'other'

        for idx, drink in enumerate(df['drink']):
            if not isinstance(drink, str):
                continue

            drink_lower = drink.lower()

            if any(term in drink_lower for term in ['coke', 'cola', 'soda', 'pepsi', 'sprite']):
                processed_df.at[idx, 'drink_category'] = 'soda'
            elif 'water' in drink_lower:
                processed_df.at[idx, 'drink_category'] = 'water'
            elif 'tea' in drink_lower:
                processed_df.at[idx, 'drink_category'] = 'tea'
            elif any(term in drink_lower for term in ['beer', 'wine', 'sake', 'alcohol']):
                processed_df.at[idx, 'drink_category'] = 'alcohol'
            elif any(term in drink_lower for term in ['juice', 'lemonade']):
                processed_df.at[idx, 'drink_category'] = 'juice'

    for col in list(processed_df.columns):
        if processed_df[col].nunique() <= 1:
            processed_df.drop(columns=[col], inplace=True)

    cols_to_keep = list(set(feature_lists['numerical_cols'] + 
                             feature_lists['categorical_cols'] + 
                             feature_lists['binary_cols']))
    
    processed_df = processed_df[cols_to_keep]

    return processed_df

def predict_food_type(input_csv_path, model_dir='logistic_regression_model'):
    """
    Predict food type for input CSV using saved model.
    
    Parameters:
    - input_csv_path (str): Path to input CSV file
    - model_dir (str, optional): Directory containing saved model files
    
    Returns:
    - numpy.ndarray: Predicted food types
    """
    with open(os.path.join(model_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    
    with open(os.path.join(model_dir, 'feature_lists.pkl'), 'rb') as f:
        feature_lists = pickle.load(f)

    input_df = pd.read_csv(input_csv_path)

    processed_input = preprocess_data(input_df, feature_lists)

    X_transformed = preprocessor.transform(processed_input)
    predictions = classifier.predict(X_transformed)

    return predictions

def evaluate_model_accuracy(input_csv_path, model_dir='logistic_regression_model'):
    """
    Evaluate model accuracy on the input CSV file.
    
    Parameters:
    - input_csv_path (str): Path to input CSV file
    - model_dir (str, optional): Directory containing saved model files
    
    Returns:
    - float: Accuracy of the model
    """
    with open(os.path.join(model_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
        classifier = pickle.load(f)
    
    with open(os.path.join(model_dir, 'feature_lists.pkl'), 'rb') as f:
        feature_lists = pickle.load(f)

    input_df = pd.read_csv(input_csv_path)

    processed_input = preprocess_data(input_df, feature_lists)

    X = processed_input
    y_true = input_df['Label'] if 'Label' in input_df.columns else input_df['food_type']

    X_transformed = preprocessor.transform(X)
    y_pred = classifier.predict(X_transformed)

    accuracy = np.mean(y_pred == y_true)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    """
    Example usage of prediction function.
    """

    input_file = 'cleaned_data_combined_modified.csv'
    try:
        predictions = predict_food_type(input_file)
        print("Predictions: ", predictions)
        # accuracy = evaluate_model_accuracy(input_file)
        # print("Accuracy: ", accuracy)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()