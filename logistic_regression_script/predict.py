import os
import sys
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter
import random

class FoodSurveyDataLoader:
    def __init__(self, csv_path):
        """
        Initialize the FoodSurveyDataLoader with the path to the CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing the food survey data
        """
        self.csv_path = csv_path
        self.df = None
        self.feature_names = []
        
        # For label encoding
        self.classes_ = None
        self.class_to_index = {}
        self.index_to_class = {}
        
        # Store column original names for easier reference
        self.q1_col = 'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'
        self.q2_col = 'Q2: How many ingredients would you expect this food item to contain?'
        self.q3_col = 'Q3: In what setting would you expect this food to be served? Please check all that apply'
        self.q4_col = 'Q4: How much would you expect to pay for one serving of this food item?'
        self.q5_col = 'Q5: What movie do you think of when thinking of this food item?'
        self.q6_col = 'Q6: What drink would you pair with this food item?'
        self.q7_col = 'Q7: When you think about this food item, who does it remind you of?'
        self.q8_col = 'Q8: How much hot sauce would you add to this food item?'
        
        # Maps for categorical variables
        self.q1_values = [1, 2, 3, 4, 5]  # Complexity scale 1-5
        self.hot_sauce_values = ['none', 'a little (mild)', 'a moderate amount (medium)', 'a lot (hot)']
        
        # Stop words for text analysis
        self.stop_words = {
            'the', 'a', 'an', 'in', 'of', 'to', 'and', 'is', 'it', 'this', 'that', 'would', 'with', 
            'for', 'on', 'at', 'be', 'i', 'food', 'item', 'my', 'me', 'you', 'your', 'they', 'their', 'or',
            'not', 'by', 'as', 'but', 'from', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'can', 
            'could', 'should', 'when', 'where', 'how', 'what', 'why', 'which', 'who', 'whom', 'whose',
            'am', 'is', 'are', 'was', 'were', 'been', 'being'
        }
        
        # Vocabulary for text columns
        self.q5_vocabulary = {}
        self.q6_vocabulary = {}
        self.q7_vocabulary = {}
        
    def load_data(self):
        """Load the CSV data"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self.df
    
    def extract_scale(self, text):
        """Extract scale rating (1-5) from Q1"""
        if pd.isna(text):
            return None
        match = re.search(r'(\d+)', str(text))
        if match:
            num = int(match.group(1))
            if 1 <= num <= 5:
                return num
        return None
    
    def extract_ingredients(self, text):
        """Extract number of ingredients from Q2"""
        if pd.isna(text):
            return None
        match = re.search(r'(\d+)', str(text))
        if match:
            return int(match.group(1))
        return str(text).count(',') + 1
    
    def text_to_number(self, text):
        """Convert text numbers to actual numbers"""
        if pd.isna(text):
            return text
        
        text = str(text).lower()
        number_dict = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
        }
        
        compound_pattern = r'(twenty|thirty|forty|fifty)[\s-](one|two|three|four|five|six|seven|eight|nine)'
        matches = re.finditer(compound_pattern, text)
        for match in matches:
            parts = re.split(r'[\s-]', match.group(0))
            if len(parts) == 2 and parts[0] in number_dict and parts[1] in number_dict:
                value = number_dict[parts[0]] + number_dict[parts[1]]
                text = text.replace(match.group(0), str(value))
        
        for word, num in number_dict.items():
            text = re.sub(r'\b' + word + r'\b', str(num), text)
        
        return text
    
    def extract_price(self, text):
        """Extract price from Q4"""
        if pd.isna(text):
            return None
        
        text = self.text_to_number(str(text).lower())

        range_match = re.search(r'(\d+[\.,]?\d*)\s*[-–—to]\s*(\d+[\.,]?\d*)', text)
        if range_match:
            try:
                low = float(range_match.group(1).replace(',', '.'))
                high = float(range_match.group(2).replace(',', '.'))
                return (low + high) / 2
            except:
                pass
        
        match = re.search(r'(\d+[\.,]?\d*)', text)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except:
                pass
        
        return None
    
    def parse_checkbox_responses(self, text):
        """Parse checkbox responses for Q3 and Q7"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # Split by common checkbox separators
        options = re.split(r'[,;|]', text)
        
        # Clean up each option
        options = [opt.strip() for opt in options if opt.strip()]
        
        return options
    
    def preprocess_text_for_bow(self, text):
        """Preprocess text for bag-of-words approach"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def encode_labels(self, labels):
        """
        Custom implementation of label encoding without sklearn
        """
        unique_labels = sorted(set(labels))
        self.classes_ = unique_labels
        self.class_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.index_to_class = {i: label for i, label in enumerate(unique_labels)}
        return labels.map(self.class_to_index)
    
    def bag_of_words(self, texts, min_doc_fraction=0.01, max_doc_fraction=0.9):
        """
        Custom implementation of bag-of-words without sklearn
        
        Parameters:
        -----------
        texts : pd.Series
            Series of text documents
        min_doc_fraction : float
            Minimum document frequency (as a fraction of total documents)
        max_doc_fraction : float
            Maximum document frequency (as a fraction of total documents)
            
        Returns:
        --------
        tuple
            (bag_of_words_df, vocabulary)
        """
        # Count words in all documents
        word_counts = Counter()
        doc_counts = Counter()
        doc_words = []
        
        # Process each document
        for doc in texts:
            if pd.isna(doc) or not isinstance(doc, str) or not doc.strip():
                doc_words.append([])
                continue
                
            # Tokenize
            words = re.findall(r'\b\w+\b', doc.lower())
            
            # Filter stop words
            words = [w for w in words if w not in self.stop_words and len(w) > 1]
            doc_words.append(words)
            
            # Count words in this document
            doc_word_set = set(words)
            for word in words:
                word_counts[word] += 1
            for word in doc_word_set:
                doc_counts[word] += 1
        
        # Filter by document frequency
        n_docs = len(texts)
        min_doc_count = max(1, int(min_doc_fraction * n_docs))
        max_doc_count = min(n_docs, int(max_doc_fraction * n_docs))
        
        # Create vocabulary
        vocabulary = {}
        feature_names = []
        for word, count in doc_counts.items():
            if min_doc_count <= count <= max_doc_count:
                vocabulary[word] = len(vocabulary)
                feature_names.append(word)
        
        # Create BoW matrix
        bow_matrix = []
        for words in doc_words:
            word_counts = Counter(words)
            row = [0] * len(vocabulary)
            for word, count in word_counts.items():
                if word in vocabulary:
                    row[vocabulary[word]] = count
            bow_matrix.append(row)
        
        # Convert to DataFrame
        bow_df = pd.DataFrame(bow_matrix, columns=feature_names)
        
        return bow_df, vocabulary
    
    def custom_one_hot_encoding(self, column, prefix):
        """
        Create one-hot encoding for a column without using pandas get_dummies
        
        Parameters:
        -----------
        column : pd.Series
            Column to one-hot encode
        prefix : str
            Prefix for the new column names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with one-hot encoded columns
        """
        # Get unique values
        unique_values = column.dropna().unique()
        
        # Create one-hot encoded columns
        one_hot_df = pd.DataFrame(index=column.index)
        for value in unique_values:
            col_name = f"{prefix}_{value}".replace(" ", "_").replace("(", "").replace(")", "").lower()
            one_hot_df[col_name] = (column == value).astype(int)
            
        return one_hot_df
    
    def stratified_train_test_split(self, X, y, test_size=0.2, random_state=None):
        """
        Custom implementation of stratified train-test split
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if random_state is not None:
            random.seed(random_state)
        
        # Convert to numpy for ease of indexing
        X_np = X.values
        y_np = y.values
        
        # Group indices by class
        class_indices = {}
        for i, label in enumerate(y_np):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        train_indices = []
        test_indices = []
        
        # For each class, split indices
        for label, indices in class_indices.items():
            n_test = int(len(indices) * test_size)
            
            # Shuffle indices
            shuffled = indices.copy()
            random.shuffle(shuffled)
            
            # Split
            test_indices.extend(shuffled[:n_test])
            train_indices.extend(shuffled[n_test:])
        
        # Create DataFrame slices
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self):
        """Preprocess the data and extract features with one-hot encoding for all categorical variables"""
        if self.df is None:
            self.load_data()
            
        # Store original column count
        original_col_count = len(self.df.columns)
        
        # =====================
        # Process Q1: Complexity (as categorical)
        # =====================
        # Extract complexity values
        self.df['Q1_complexity_raw'] = self.df[self.q1_col].apply(self.extract_scale)
        
        # One-hot encode complexity (1-5)
        for value in self.q1_values:
            col_name = f'Q1_complexity_{value}'
            self.df[col_name] = (self.df['Q1_complexity_raw'] == value).astype(int)
        
        # =====================
        # Process Q2: Ingredients (numeric)
        # =====================
        self.df['Q2_ingredients'] = self.df[self.q2_col].apply(self.extract_ingredients)
        
        # =====================
        # Process Q3: Settings (categorical from checkbox)
        # =====================
        # Extract all unique settings from the data
        q3_all_settings = set()
        for text in self.df[self.q3_col]:
            settings = self.parse_checkbox_responses(text)
            q3_all_settings.update(settings)
        
        # One-hot encode each setting
        for setting in q3_all_settings:
            col_name = f'Q3_setting_{setting.replace(" ", "_").lower()}'
            self.df[col_name] = self.df[self.q3_col].apply(
                lambda x: 1 if setting in self.parse_checkbox_responses(x) else 0
            )
        
        # =====================
        # Process Q4: Price (numeric)
        # =====================
        self.df['Q4_price'] = self.df[self.q4_col].apply(self.extract_price)
        
        # =====================
        # Process Q5: Movie (text - bag of words)
        # =====================
        # Preprocess text
        self.df['Q5_movie_text'] = self.df[self.q5_col].apply(self.preprocess_text_for_bow)
        
        # Apply bag-of-words using custom implementation
        q5_bow_df, self.q5_vocabulary = self.bag_of_words(
            self.df['Q5_movie_text'], 
            min_doc_fraction=0.01, 
            max_doc_fraction=0.9
        )
        
        # Add prefix to column names
        q5_bow_df.columns = [f'Q5_movie_{col}' for col in q5_bow_df.columns]
        
        # Join with main DataFrame
        self.df = pd.concat([self.df, q5_bow_df], axis=1)
        
        # =====================
        # Process Q6: Drink (text - bag of words)
        # =====================
        # Preprocess text
        self.df['Q6_drink_text'] = self.df[self.q6_col].apply(self.preprocess_text_for_bow)
        
        # Apply bag-of-words using custom implementation
        q6_bow_df, self.q6_vocabulary = self.bag_of_words(
            self.df['Q6_drink_text'], 
            min_doc_fraction=0.01, 
            max_doc_fraction=0.9
        )
        
        # Add prefix to column names
        q6_bow_df.columns = [f'Q6_drink_{col}' for col in q6_bow_df.columns]
        
        # Join with main DataFrame
        self.df = pd.concat([self.df, q6_bow_df], axis=1)
        
        # =====================
        # Process Q7: Reminds of (text - bag of words)
        # =====================
        # Preprocess text
        self.df['Q7_reminds_text'] = self.df[self.q7_col].apply(self.preprocess_text_for_bow)
        
        # Apply bag-of-words using custom implementation
        q7_bow_df, self.q7_vocabulary = self.bag_of_words(
            self.df['Q7_reminds_text'], 
            min_doc_fraction=0.01, 
            max_doc_fraction=0.9
        )
        
        # Add prefix to column names
        q7_bow_df.columns = [f'Q7_reminds_{col}' for col in q7_bow_df.columns]
        
        # Join with main DataFrame
        self.df = pd.concat([self.df, q7_bow_df], axis=1)
        
        # =====================
        # Process Q8: Hot sauce (categorical)
        # =====================
        hot_sauce_df = self.custom_one_hot_encoding(self.df[self.q8_col], 'Q8_hot_sauce')
        self.df = pd.concat([self.df, hot_sauce_df], axis=1)
        
        # Print original row count
        original_row_count = len(self.df)
        
        # Get count of missing values before imputation
        missing_numeric_mask = self.df['Q2_ingredients'].isna() | self.df['Q4_price'].isna()
        missing_count = missing_numeric_mask.sum()
        
        if missing_count > 0:
            print(f"Found {missing_count} rows with missing numeric values ({missing_count/original_row_count:.1%} of data)")
            print("Imputing missing values instead of dropping rows")
            
            self.df['Q2_ingredients'].fillna(5.0, inplace=True)
            
            self.df['Q4_price'].fillna(10.0, inplace=True)
            
            # Verify no missing values remain in numeric columns
            remaining_missing = self.df['Q2_ingredients'].isna() | self.df['Q4_price'].isna()
            if remaining_missing.sum() > 0:
                print(f"Warning: {remaining_missing.sum()} rows still have missing numeric values after imputation")
            else:
                print("Successfully imputed all missing numeric values")
        
        exclude_cols = [
            'Label', 'label_encoded',
            self.q1_col, self.q2_col, self.q3_col, self.q4_col, 
            self.q5_col, self.q6_col, self.q7_col, self.q8_col,
            'Q1_complexity_raw', 'Q5_movie_text', 'Q6_drink_text', 'Q7_reminds_text'
        ]
        
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]
        
        if 'Label' in self.df.columns:
            self.df['label_encoded'] = self.encode_labels(self.df['Label'])
        
        print(f"Preprocessed data with {len(self.feature_names)} features")
        print(f"Started with {original_col_count} columns, ended with {len(self.df.columns)} columns")
        print(f"Data shape after preprocessing: {self.df.shape}")
        
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, feature_names)
        """
        if self.feature_names is None or len(self.feature_names) == 0:
            self.preprocess_data()
        
        X = self.df[self.feature_names]
        y = self.df['label_encoded']
        
        X_train, X_test, y_train, y_test = self.stratified_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Split data into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
        print(f"Feature matrix shape: {X.shape}")
        
        return X_train, X_test, y_train, y_test, self.feature_names, self
    
    def get_feature_groups(self):
        """
        Get feature names grouped by question.
        
        Returns:
        --------
        dict
            Dictionary of feature groups by question
        """
        feature_groups = {
            'Q1_complexity': [col for col in self.feature_names if col.startswith('Q1_complexity_')],
            'Q2_ingredients': ['Q2_ingredients'],
            'Q3_setting': [col for col in self.feature_names if col.startswith('Q3_setting_')],
            'Q4_price': ['Q4_price'],
            'Q5_movie': [col for col in self.feature_names if col.startswith('Q5_movie_')],
            'Q6_drink': [col for col in self.feature_names if col.startswith('Q6_drink_')],
            'Q7_reminds': [col for col in self.feature_names if col.startswith('Q7_reminds_')],
            'Q8_hot_sauce': [col for col in self.feature_names if col.startswith('Q8_hot_sauce_')]
        }
        
        # Print feature group summary
        for group, features in feature_groups.items():
            print(f"{group}: {len(features)} features")
            
        return feature_groups


class FeatureColumnMismatchHandler:
    """
    This class handles the mismatch between expected model features and features in test data.
    It adds missing columns with default values (0) and removes extra columns.
    """
    
    def __init__(self, expected_columns):
        """
        Initialize with the list of expected feature columns.
        
        Parameters:
        -----------
        expected_columns : list
            List of column names expected by the model
        """
        self.expected_columns = expected_columns
    
    def transform(self, X):
        """
        Transform the input DataFrame to match the expected columns.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input feature matrix
            
        Returns:
        --------
        pandas.DataFrame
            Transformed feature matrix with matching columns
        """
        # Find missing columns
        missing_cols = [col for col in self.expected_columns if col not in X.columns]
        
        # Add missing columns with default values (0)
        for col in missing_cols:
            X[col] = 0
            
        # Make sure we only keep the expected columns in the correct order
        return X[self.expected_columns]


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
    
    # Extract the expected feature columns from the preprocessor or feature lists
    if hasattr(preprocessor, 'feature_names_in_'):
        expected_columns = preprocessor.feature_names_in_
    else:
        # Fallback: try to get expected columns from feature_lists
        expected_columns = feature_lists.get('features', [])
    
    return preprocessor, classifier, feature_lists, expected_columns

def predict_all(filename):
    """
    Make predictions for all examples in the given CSV file.
    
    Args:
        filename: Path to the CSV file containing the test data
        
    Returns:
        A list of predictions (food type labels) for each example
    """
    try:
        preprocessor, classifier, feature_lists, expected_columns = load_model_components()
        
        dataloader = FoodSurveyDataLoader(filename)
        dataloader.load_data()
        
        processed_data = dataloader.preprocess_data()
        
        X = processed_data[dataloader.feature_names]
        
        # Handle feature mismatch
        print("Handling feature column mismatch...")
        missing_cols = [col for col in expected_columns if col not in X.columns]
        if missing_cols:
            print(f"Adding {len(missing_cols)} missing columns with zeros")
            feature_handler = FeatureColumnMismatchHandler(expected_columns)
            X_aligned = feature_handler.transform(X)
        else:
            X_aligned = X[expected_columns]
        
        X_processed = preprocessor.transform(X_aligned)
        pred_encoded = classifier.predict(X_processed)
        
        # Map predictions back to labels
        index_to_class = {0: 'Pizza', 1: 'Shawarma', 2: 'Sushi'}
        if hasattr(dataloader, 'index_to_class') and dataloader.index_to_class:
            index_to_class = dataloader.index_to_class
        
        predictions = [index_to_class.get(code, 'Unknown') for code in pred_encoded]
        
        return predictions
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        # In case of error, return a fallback prediction for each example
        try:
            test_data = pd.read_csv(filename)
            return ['Pizza'] * len(test_data)
        except:
            return []

if __name__ == "__main__":
    # filename = "cleaned_data_combined_modified.csv"  # Pizza, Shawarma, Sushi, Pizza, Shawarma, Sushi
    filename = "example_data.csv"  # Pizza, Shawarma, Sushi, Pizza, Shawarma, Sushi
    
    # Make predictions
    predictions = predict_all(filename)
    
    print(f"Made {len(predictions)} predictions:")
    for i, pred in enumerate(predictions):  # Print all predictions
        print(f"Example {i+1}: {pred}")