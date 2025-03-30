import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

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
        self.label_encoder = LabelEncoder()
        
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
        
        # Initialize vectorizers for text columns
        self.q5_vectorizer = None  # Will be initialized in preprocess_data
        self.q6_vectorizer = None
        self.q7_vectorizer = None
        
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
        
        # Apply bag-of-words using CountVectorizer
        self.q5_vectorizer = CountVectorizer(
            min_df=0.01,  # Minimum document frequency (at least 1% of documents)
            max_df=0.9,   # Maximum document frequency (at most 90% of documents)
            stop_words=list(self.stop_words)
        )
        
        # Transform text to bag-of-words features
        q5_bow_features = self.q5_vectorizer.fit_transform(self.df['Q5_movie_text'])
        
        # Convert to DataFrame and add prefix
        q5_bow_df = pd.DataFrame(
            q5_bow_features.toarray(),
            columns=[f'Q5_movie_{word}' for word in self.q5_vectorizer.get_feature_names_out()]
        )
        
        # Join with main DataFrame
        self.df = pd.concat([self.df, q5_bow_df], axis=1)
        
        # =====================
        # Process Q6: Drink (text - bag of words)
        # =====================
        # Preprocess text
        self.df['Q6_drink_text'] = self.df[self.q6_col].apply(self.preprocess_text_for_bow)
        
        # Apply bag-of-words using CountVectorizer
        self.q6_vectorizer = CountVectorizer(
            min_df=0.01,  # Minimum document frequency
            max_df=0.9,   # Maximum document frequency
            stop_words=list(self.stop_words)
        )
        
        # Transform text to bag-of-words features
        q6_bow_features = self.q6_vectorizer.fit_transform(self.df['Q6_drink_text'])
        
        # Convert to DataFrame and add prefix
        q6_bow_df = pd.DataFrame(
            q6_bow_features.toarray(),
            columns=[f'Q6_drink_{word}' for word in self.q6_vectorizer.get_feature_names_out()]
        )
        
        # Join with main DataFrame
        self.df = pd.concat([self.df, q6_bow_df], axis=1)
        
        # =====================
        # Process Q7: Reminds of (text - bag of words)
        # =====================
        # Preprocess text
        self.df['Q7_reminds_text'] = self.df[self.q7_col].apply(self.preprocess_text_for_bow)
        
        # Apply bag-of-words using CountVectorizer
        self.q7_vectorizer = CountVectorizer(
            min_df=0.01,  # Minimum document frequency
            max_df=0.9,   # Maximum document frequency
            stop_words=list(self.stop_words)
        )
        
        # Transform text to bag-of-words features
        q7_bow_features = self.q7_vectorizer.fit_transform(self.df['Q7_reminds_text'])
        
        # Convert to DataFrame and add prefix
        q7_bow_df = pd.DataFrame(
            q7_bow_features.toarray(),
            columns=[f'Q7_reminds_{word}' for word in self.q7_vectorizer.get_feature_names_out()]
        )
        
        # Join with main DataFrame
        self.df = pd.concat([self.df, q7_bow_df], axis=1)
        
        # =====================
        # Process Q8: Hot sauce (categorical)
        # =====================
        # One-hot encode hot sauce directly
        hot_sauce_dummies = pd.get_dummies(self.df[self.q8_col], prefix='Q8_hot_sauce')
        self.df = pd.concat([self.df, hot_sauce_dummies], axis=1)
        
        # =====================
        # Drop rows with missing numeric values
        # =====================
        # Print original row count
        original_row_count = len(self.df)
        
        # Drop rows with missing values in Q2_ingredients and Q4_price
        missing_numeric_mask = self.df['Q2_ingredients'].isna() | self.df['Q4_price'].isna()
        rows_to_drop = missing_numeric_mask.sum()
        
        if rows_to_drop > 0:
            self.df = self.df[~missing_numeric_mask]
            print(f"Dropped {rows_to_drop} rows with missing numeric values ({rows_to_drop/original_row_count:.1%} of data)")
        
        # =====================
        # Prepare feature list for model training
        # =====================
        # Identify feature columns for model training (exclude raw text and intermediate columns)
        exclude_cols = [
            'Label', 'label_encoded',
            self.q1_col, self.q2_col, self.q3_col, self.q4_col, 
            self.q5_col, self.q6_col, self.q7_col, self.q8_col,
            'Q1_complexity_raw', 'Q5_movie_text', 'Q6_drink_text', 'Q7_reminds_text'
        ]
        
        # Get all the feature columns
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]
        
        # Encode labels
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['Label'])
        
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
            (X_train, X_test, y_train, y_test, feature_names, label_encoder)
        """
        from sklearn.model_selection import train_test_split
        
        if self.feature_names is None or len(self.feature_names) == 0:
            self.preprocess_data()
        
        # Get features and target
        X = self.df[self.feature_names]
        y = self.df['label_encoded']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Split data into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
        print(f"Feature matrix shape: {X.shape}")
        
        return X_train, X_test, y_train, y_test, self.feature_names, self.label_encoder
    
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