import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import joblib

def extract_number(value):
    """Extract numerical values from strings."""
    if pd.isna(value) or not isinstance(value, str):
        return value

    # Try to find numbers in the string
    matches = re.findall(r'(\d+(\.\d+)?)', value)
    if matches and len(matches) > 0:
        return float(matches[0][0])
    return np.nan

def preprocess_data_for_logistic_regression(file_path):
    """
    Preprocess food survey data specifically for logistic regression.

    This function:
    1. Cleans and extracts numerical values
    2. Creates appropriate features for logistic regression
    3. Handles categorical data properly
    4. Returns both raw features and a preprocessing pipeline
    """
    # Load data
    print(f"Loading data from {file_path}")
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

    # Rename columns if they exist
    for old_col, new_col in columns_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Create separate dataframe for feature engineering
    processed_df = pd.DataFrame(index=df.index)

    # Extract numerical features
    if 'complexity' in df.columns:
        processed_df['complexity'] = df['complexity'].copy()

    if 'ingredients' in df.columns:
        processed_df['ingredients_num'] = df['ingredients'].apply(extract_number)

    if 'price' in df.columns:
        processed_df['price_num'] = df['price'].apply(extract_number)

    # Process hot sauce as ordinal feature
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

    # Process movie data - apply bag of words approach instead of manually categorizing
    if 'movie' in df.columns:
        # Clean movie text data
        df['movie_cleaned'] = df['movie'].copy()
        df.loc[df['movie_cleaned'].isna(), 'movie_cleaned'] = ''  # Replace NaN with empty string
        
        # Apply bag of words using CountVectorizer
        # Set min_df to exclude very rare words and max_features to limit vocabulary size
        movie_vectorizer = CountVectorizer(max_features=20, min_df=3, 
                                           stop_words='english', binary=True)
        
        # Fit and transform on non-empty strings only
        non_empty_movies = df['movie_cleaned'].fillna('').astype(str)
        non_empty_indices = non_empty_movies.str.len() > 0
        
        if sum(non_empty_indices) > 0:  # Only if we have non-empty values
            # Apply vectorizer to create bag of words features
            movie_features = movie_vectorizer.fit_transform(non_empty_movies[non_empty_indices])
            
            # Create new columns for each word
            movie_df = pd.DataFrame(movie_features.toarray(), 
                                   columns=[f'movie_word_{word}' for word in movie_vectorizer.get_feature_names_out()],
                                   index=df.index[non_empty_indices])
            
            # Merge these features into the processed dataframe
            for col in movie_df.columns:
                processed_df.loc[movie_df.index, col] = movie_df[col]
                # Fill remaining rows with 0
                processed_df[col] = processed_df[col].fillna(0).astype(int)
                
            print(f"Created {len(movie_df.columns)} bag-of-words features for movie text")
        else:
            print("No non-empty movie data to create bag-of-words features")

    # Process drink preferences - create drink type categorization
    if 'drink' in df.columns:
        processed_df['drink_category'] = 'other'  # Default

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

    # Add the target variable
    if 'food_type' in df.columns:
        processed_df['food_type'] = df['food_type']

    # Define numerical and categorical columns
    numerical_cols = [col for col in ['complexity', 'ingredients_num', 'price_num', 'hot_sauce_level']
                     if col in processed_df.columns]

    # Get all movie word features
    movie_word_cols = [col for col in processed_df.columns if col.startswith('movie_word_')]
    
    # Now we keep the movie word columns separate from other categorical columns
    categorical_cols = [col for col in processed_df.columns
                       if col.startswith('setting_') or
                       col == 'drink_category']

    # Check for and remove constant columns
    for cols_list in [numerical_cols, categorical_cols, movie_word_cols]:
        for col in list(cols_list):  # Use list() to avoid modifying during iteration
            if col in processed_df.columns and processed_df[col].nunique() <= 1:
                print(f"Removing column {col} because it has only one unique value")
                cols_list.remove(col)

    print(f"Numerical features: {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")
    print(f"Movie word features: {movie_word_cols}")

    # Create preprocessing pipeline for logistic regression
    # We'll handle movie word features as binary features without additional preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols),
            # Movie word features are already binary, so we just pass them through
            ('movie_words', 'passthrough', movie_word_cols)
        ])

    # Print feature counts
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    print(f"Number of movie word features: {len(movie_word_cols)}")

    return processed_df, preprocessor, numerical_cols, categorical_cols, movie_word_cols

def train_logistic_regression(df, preprocessor, numerical_cols, categorical_cols, movie_word_cols, target_col='food_type', test_size=0.25, random_state=42):
    """Train a logistic regression model with the preprocessed data."""
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create pipeline with preprocessing and logistic regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    # Define hyperparameters to tune
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__penalty': ['l1', 'l2', 'elasticnet'],
        'classifier__solver': ['saga'],  # saga supports all penalties
        'classifier__l1_ratio': [0.2, 0.5, 0.8]  # only used with elasticnet
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )

    print("Training logistic regression model...")
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    # Print results
    print("\nLogistic Regression Model Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Training Accuracy: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {best_model.score(X_test, y_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('logistic_regression_confusion_matrix.png')

    # Analyze coefficients if it's a linear model
    analyze_logistic_regression_coefficients(best_model, X, numerical_cols, categorical_cols, movie_word_cols)

    return best_model

def analyze_logistic_regression_coefficients(model, X, numerical_cols, categorical_cols, movie_word_cols):
    """Analyze and visualize the coefficients of the logistic regression model."""
    # Extract the logistic regression model from the pipeline
    logistic_model = model.named_steps['classifier']

    # Get the preprocessor (already fitted during training)
    preprocessor = model.named_steps['preprocessor']

    # Get feature names from the fitted preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Get the coefficients for each class
    coefficients = pd.DataFrame(
        logistic_model.coef_,
        columns=feature_names,
        index=logistic_model.classes_
    )

    # For each class, get the top features
    plt.figure(figsize=(15, 10))

    for i, food_type in enumerate(logistic_model.classes_):
        plt.subplot(len(logistic_model.classes_), 1, i+1)

        # Sort coefficients for this class
        sorted_coef = coefficients.loc[food_type].sort_values()

        # Plot top and bottom 10 features
        n_features = min(10, len(sorted_coef) // 2)
        top_features = sorted_coef.tail(n_features)
        bottom_features = sorted_coef.head(n_features)

        features_to_plot = pd.concat([bottom_features, top_features])

        # Plot horizontal bar chart
        plt.barh(range(len(features_to_plot)), features_to_plot.values)
        plt.yticks(range(len(features_to_plot)), features_to_plot.index)
        plt.title(f'Top Features for {food_type}')
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients.png')

    # Print top features for each class
    print("\nTop Features by Food Type:")
    for food_type in logistic_model.classes_:
        print(f"\n{food_type}:")
        top_positive = coefficients.loc[food_type].nlargest(5)
        top_negative = coefficients.loc[food_type].nsmallest(5)

        print("Positive indicators:")
        for feature, coef in top_positive.items():
            print(f"  {feature}: {coef:.4f}")

        print("Negative indicators:")
        for feature, coef in top_negative.items():
            print(f"  {feature}: {coef:.4f}")

def main():
    # File path to your CSV
    file_path = 'cleaned_data_combined_modified.csv'

    # Preprocess data for logistic regression
    print("Preprocessing data for logistic regression...")
    df, preprocessor, numerical_cols, categorical_cols, movie_word_cols = preprocess_data_for_logistic_regression(file_path)

    # Train logistic regression model
    model = train_logistic_regression(df, preprocessor, numerical_cols, categorical_cols, movie_word_cols)

    # Save model and preprocessor
    output_dir = "logistic_regression_model"
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, f"{output_dir}/model.pkl")

    print(f"Model saved to {output_dir}/model.pkl")
    print("You can use this model for predictions on new data.")

if __name__ == "__main__":
    main()