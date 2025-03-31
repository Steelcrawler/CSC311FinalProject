import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from very_good_dataloader import FoodSurveyDataLoader

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
    df = pd.read_csv(file_path)

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

    if 'complexity' in df.columns:
        processed_df['complexity'] = df['complexity'].copy()

    if 'ingredients' in df.columns:
        processed_df['ingredients_num'] = df['ingredients'].apply(extract_number)

    if 'price' in df.columns:
        processed_df['price_num'] = df['price'].apply(extract_number)

    if 'hot_sauce' in df.columns:
        hot_sauce_map = {
            'None': 0,
            'A little (mild)': 1,
            'A moderate amount (medium)': 2,
            'A lot (hot)': 3,
            'I will have some of this food item with my hot sauce': 4
        }
        processed_df['hot_sauce_level'] = df['hot_sauce'].map(hot_sauce_map).fillna(-1)

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

    if 'food_type' in df.columns:
        processed_df['food_type'] = df['food_type']

    for col in list(processed_df.columns):
        if processed_df[col].nunique() <= 1:
            print(f"Removing column {col} because it has only one unique value")
            processed_df.drop(columns=[col], inplace=True)

    X = processed_df.drop(columns=['food_type'])
    y = processed_df['food_type']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    binary_cols = []
    for col in X.columns:
        unique_values = X[col].unique()
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False}):
            binary_cols.append(col)

    return X, y, numerical_cols, categorical_cols, binary_cols

def train_with_grid_search(file_path, output_dir='logistic_regression_model'):
    """
    Train logistic regression model with grid search for hyperparameter tuning.
    Create simple visualizations of validation accuracy vs hyperparameters.
    """
    dataloader = FoodSurveyDataLoader(file_path)
    dataloader.load_data()
    df = dataloader.preprocess_data()

    X = df[dataloader.feature_names]
    y = df['label_encoded']

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Get feature groups to determine numerical and binary columns
    feature_groups = dataloader.get_feature_groups()

    numerical_cols = (
        feature_groups['Q2_ingredients'] +
        feature_groups['Q4_price'] +
        feature_groups['Q5_movie'] +
        feature_groups['Q6_drink'] +
        feature_groups['Q7_reminds']
    )

    binary_cols = (
        feature_groups['Q1_complexity'] +
        feature_groups['Q3_setting'] +
        feature_groups['Q8_hot_sauce']
    )

    # Configure preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('binary', 'passthrough', binary_cols)
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Simplified parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': range(40),
    }

    results = {
        'C': [],
        'max_iter': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'test_accuracy': []
    }

    print(f"Testing {len(param_grid['C']) * len(param_grid['max_iter'])} parameter combinations")

    best_score = 0
    best_params = None
    best_model = None

    os.makedirs(output_dir, exist_ok=True)

    for C in param_grid['C']:
        for max_iter in param_grid['max_iter']:
            try:
                model = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=42
                )
                model.fit(X_train_processed, y_train)
                
                y_train_pred = model.predict(X_train_processed)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                
                y_val_pred = model.predict(X_val_processed)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                
                X_test_processed = preprocessor.transform(X_test)
                y_test_pred = model.predict(X_test_processed)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                results['C'].append(C)
                results['max_iter'].append(max_iter)
                results['train_accuracy'].append(train_accuracy)
                results['val_accuracy'].append(val_accuracy)
                results['test_accuracy'].append(test_accuracy)
                
                print(f"Params: C={C}, max_iter={max_iter}, Train acc: {train_accuracy:.4f}, Val acc: {val_accuracy:.4f}, Test acc: {test_accuracy:.4f}")
                
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = {'C': C, 'max_iter': max_iter}
                    best_model = model
                    
            except Exception as e:
                print(f"Error with params C={C}, max_iter={max_iter}: {e}")

    results_df = pd.DataFrame(results)

    create_hyperparameter_plots(results_df, output_dir)

    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            random_state=42
        ))
    ])

    best_pipeline.fit(X_train, y_train)

    y_test_pred = best_pipeline.predict(X_test)

    print("\nBest Model Parameters (selected based on validation accuracy):")
    print(f"C: {best_params['C']}")
    print(f"max_iter: {best_params['max_iter']}")

    print("\nTest Classification Report for Best Model:")
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix([dataloader.index_to_class[y] for y in y_test], [dataloader.index_to_class[y] for y in y_test_pred])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pizza', 'Sushi', 'Shawarma'], 
                yticklabels=['Pizza', 'Sushi', 'Shawarma'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_test.png")

    with open(f"{output_dir}/preprocessor.pkl", 'wb') as f:
        pickle.dump(best_pipeline.named_steps['preprocessor'], f)

    with open(f"{output_dir}/classifier.pkl", 'wb') as f:
        pickle.dump(best_pipeline.named_steps['classifier'], f)

    feature_dict = {
        'numerical_cols': numerical_cols,
        'binary_cols': binary_cols
    }

    with open(f"{output_dir}/feature_lists.pkl", 'wb') as f:
        pickle.dump(feature_dict, f)

    results_df.to_csv(f"{output_dir}/grid_search_results.csv", index=False)

    print(f"\nModel components saved in {output_dir}")
    print("Files saved:")
    print("- preprocessor.pkl")
    print("- classifier.pkl")
    print("- feature_lists.pkl")
    print("- grid_search_results.csv")
    print("- Simple 2D plots in the output directory")

    return best_pipeline, results_df

def create_hyperparameter_plots(results_df, output_dir):
    """
    Create simple 2D plots to visualize the effect of hyperparameters on validation accuracy.
    """
    plt.figure(figsize=(10, 6))
    c_vs_val = results_df.groupby('C')['val_accuracy'].mean().reset_index()
    plt.plot(c_vs_val['C'], c_vs_val['val_accuracy'], marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of C on Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_vs_validation_accuracy.png")

    # Plot 2: max_iter vs. Validation Accuracy
    plt.figure(figsize=(10, 6))
    iter_vs_val = results_df.groupby('max_iter')['val_accuracy'].mean().reset_index()
    plt.plot(iter_vs_val['max_iter'], iter_vs_val['val_accuracy'], marker='o', linestyle='-')
    plt.xlabel('Maximum Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of Maximum Iterations on Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/max_iter_vs_validation_accuracy.png")

    # Plot 3: C vs. Validation Accuracy for different max_iter values
    plt.figure(figsize=(12, 8))
    for max_iter in sorted(results_df['max_iter'].unique()):
        subset = results_df[results_df['max_iter'] == max_iter]
        if not subset.empty:
            c_vs_val_subset = subset.groupby('C')['val_accuracy'].mean().reset_index()
            plt.plot(c_vs_val_subset['C'], c_vs_val_subset['val_accuracy'], marker='o', linestyle='-', 
                    label=f'max_iter={max_iter}')
    plt.xscale('log')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of C on Validation Accuracy by max_iter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_vs_validation_accuracy_by_max_iter.png")

    # Plot 4: Heatmap of C vs max_iter on validation accuracy
    plt.figure(figsize=(12, 8))
    pivot_table = results_df.pivot_table(values='val_accuracy', index='max_iter', columns='C', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Validation Accuracy by C and max_iter')
    plt.ylabel('max_iter')
    plt.xlabel('C')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_max_iter_validation_heatmap.png")

    # Plot 5: Validation vs Train accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['train_accuracy'], results_df['val_accuracy'], 
                alpha=0.7, s=50, c=np.log10(results_df['C']), cmap='viridis')
    plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5)
    plt.colorbar(label='log10(C)')
    plt.xlabel('Training Accuracy')
    plt.ylabel('Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/train_vs_validation_accuracy.png")

def main():
    file_path = 'cleaned_data_combined_modified.csv'
    train_with_grid_search(file_path)

if __name__ == "__main__":
    main()