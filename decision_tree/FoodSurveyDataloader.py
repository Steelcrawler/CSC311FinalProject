import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

class FoodSurveyDataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self):
        """Load and process the CSV data"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self.df
    
    def extract_scale(self, text):
        """Extract scale rating from Q1"""
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
        """Parse checkbox responses for Q3"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        # Split by common checkbox separators
        options = re.split(r'[,;|]', text)
        
        # Clean up each option
        options = [opt.strip() for opt in options if opt.strip()]
        
        return options
    
    def parse_hot_sauce_amount(self, text, count_nan_as_none=True):
        """Parse hot sauce amount from Q8"""
        if pd.isna(text) or not isinstance(text, str):
            return 'none' if count_nan_as_none else None
        
        text = text.strip().lower()
        
        # Direct mapping for the specific values in the dataset
        hot_sauce_mapping = {
            'none': 'none',
            'a little (mild)': 'little',
            'a moderate amount (medium)': 'medium',
            'a lot (hot)': 'lot'
        }
        
        # Look for exact matches
        for original, category in hot_sauce_mapping.items():
            if original.lower() == text:
                return category
        
        # Fallback for partial matches
        if 'none' in text:
            return 'none'
        elif 'little' in text or 'mild' in text:
            return 'little'
        elif 'lot' in text or 'hot' in text:
            return 'lot'
        elif 'moderate' in text or 'medium' in text:
            return 'medium'
                
        return 'other'
    
    def preprocess_data(self):
        """Preprocess the data and extract features for machine learning"""
        if self.df is None:
            self.load_data()
            
        # Store original column count
        original_col_count = len(self.df.columns)
        
        # Extract numerical features
        self.df['Q1_complexity'] = self.df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].apply(self.extract_scale)
        self.df['Q2_ingredients'] = self.df['Q2: How many ingredients would you expect this food item to contain?'].apply(self.extract_ingredients)
        self.df['Q4_price'] = self.df['Q4: How much would you expect to pay for one serving of this food item?'].apply(self.extract_price)
        
        # Process categorical features
        # Q3 - Settings as multiple one-hot columns
        q3_all_settings = set()
        for text in self.df['Q3: In what setting would you expect this food to be served? Please check all that apply']:
            settings = self.parse_checkbox_responses(text)
            q3_all_settings.update(settings)
        
        # Create one-hot features for settings
        for setting in q3_all_settings:
            col_name = f'Q3_setting_{setting.replace(" ", "_").lower()}'
            self.df[col_name] = self.df['Q3: In what setting would you expect this food to be served? Please check all that apply'].apply(
                lambda x: 1 if setting in self.parse_checkbox_responses(x) else 0
            )
        
        # Q8 - Hot sauce as categorical - treat directly as categorical without intermediate parsing
        hot_sauce_dummies = pd.get_dummies(
            self.df['Q8: How much hot sauce would you add to this food item?'],
            prefix='Q8_hot_sauce'
        )
        self.df = pd.concat([self.df, hot_sauce_dummies], axis=1)
        
        # Identify numerical and categorical feature columns
        numerical_features = ['Q1_complexity', 'Q2_ingredients', 'Q4_price']
        setting_columns = [col for col in self.df.columns if col.startswith('Q3_setting_')]
        hot_sauce_columns = [col for col in self.df.columns if col.startswith('Q8_hot_sauce_')]
        
        # Check which columns have missing values among the numerical features
        missing_cols = []
        for col in numerical_features:
            if col in self.df.columns and self.df[col].isna().any():
                missing_cols.append(col)
        
        # Drop columns with missing values
        if missing_cols:
            self.df = self.df.drop(columns=missing_cols)
            print(f"Dropped {len(missing_cols)} columns with missing values: {missing_cols}")
        
        # Update feature lists after potential column dropping
        numerical_features = [col for col in numerical_features if col in self.df.columns]
        
        # Update feature names - include only numerical, settings, and hot sauce columns (no original text columns)
        self.feature_names = numerical_features + setting_columns + hot_sauce_columns
        
        # Initialize label encoder if needed
        if not hasattr(self, 'label_encoder'):
            self.label_encoder = LabelEncoder()
            
        # Encode labels
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['Label'])
        
        print(f"Preprocessed data with {len(self.feature_names)} features")
        print(f"Started with {original_col_count} columns, ended with {len(self.df.columns)} columns")
        return self.df