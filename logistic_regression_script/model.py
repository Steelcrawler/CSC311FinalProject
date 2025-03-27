import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
import os

def extract_number(value):
    """Extract numerical values from strings."""
    if pd.isna(value) or not isinstance(value, str):
        return value

    matches = re.findall(r'(\d+(\.\d+)?)', value)
    if matches and len(matches) > 0:
        return float(matches[0][0])
    return np.nan

def preprocess_data_for_logistic_regression(file_path):
    """
    Preprocess food survey data for logistic regression.
    Returns processed dataframe and feature lists.
    """
    # Load data
    df = pd.read_csv(file_path)

    # Rename columns to simpler names
    columns_mapping = {
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'complexity',
        'Q2: How many ingredients would you expect this food item to contain?': 'ingredients',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'settings',
        'Q4: How much would you expect to pay for one serving of this food item?': 'price',
        'Q5: What movie do you think of when thinking of this food item?': 'movie',
        'Q6: What drink would you pair with this food item?': 'drink',
        'Q7: When you think about this food item, who does it remind you of?': 'reminds_of',
        'Q8: How much hot sauce would you add to this food item?': 'hot_sauce',
        'Label': 'food_type'
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

    # Process settings (multiple choice) - create individual binary features for each setting
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

    # Process movie data - simple word frequency approach
    if 'movie' in df.columns:
        # Clean movie text data
        df['movie_cleaned'] = df['movie'].fillna('').str.lower()
        
        # Get word frequencies
        movie_words = df['movie_cleaned'].str.split(expand=True).stack()
        top_movie_words = movie_words.value_counts().head(20).index

        # Create binary features for top words
        for word in top_movie_words:
            col_name = f'movie_word_{word}'
            processed_df[col_name] = df['movie_cleaned'].apply(
                lambda x: 1 if word in x else 0
            )

    # Drink categorization
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

    # Add food type
    if 'food_type' in df.columns:
        processed_df['food_type'] = df['food_type']

    # Remove features with only one unique value
    for col in list(processed_df.columns):
        if processed_df[col].nunique() <= 1:
            print(f"Removing column {col} because it has only one unique value")
            processed_df.drop(columns=[col], inplace=True)

    # Separate features and target
    X = processed_df.drop(columns=['food_type'])
    y = processed_df['food_type']

    # Identify feature types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Improved binary column selection
    binary_cols = []
    for col in X.columns:
        unique_values = X[col].unique()
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False}):
            binary_cols.append(col)

    return X, y, numerical_cols, categorical_cols, binary_cols

def train_and_save_model(file_path, output_dir='logistic_regression_model'):
    """
    Train logistic regression model and save model artifacts using pickle.
    """
    # Preprocess data
    X, y, numerical_cols, categorical_cols, binary_cols = preprocess_data_for_logistic_regression(file_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ])

    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model components using pickle
    with open(f"{output_dir}/preprocessor.pkl", 'wb') as f:
        pickle.dump(pipeline.named_steps['preprocessor'], f)
    
    with open(f"{output_dir}/classifier.pkl", 'wb') as f:
        pickle.dump(pipeline.named_steps['classifier'], f)

    # Save feature lists using pickle
    feature_dict = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'binary_cols': binary_cols
    }
    
    with open(f"{output_dir}/feature_lists.pkl", 'wb') as f:
        pickle.dump(feature_dict, f)

    print(f"\nModel components saved in {output_dir}")
    print("Files saved:")
    print("- preprocessor.pkl")
    print("- classifier.pkl")
    print("- feature_lists.pkl")

def main():
    file_path = 'cleaned_data_combined_modified.csv'
    train_and_save_model(file_path)

if __name__ == "__main__":
    main()