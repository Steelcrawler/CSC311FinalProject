import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class FoodSurveyDataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
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
            'a lot (hot)': 'lot',
            'a moderate amount (medium)': 'medium'
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
        
        # Extract numerical features
        self.df['Q1_complexity'] = self.df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].apply(self.extract_scale)
        self.df['Q2_ingredients'] = self.df['Q2: How many ingredients would you expect this food item to contain?'].apply(self.extract_ingredients)
        self.df['Q4_price'] = self.df['Q4: How much would you expect to pay for one serving of this food item?'].apply(self.extract_price)
        
        # Fill missing values for numerical features with median
        for col in ['Q1_complexity', 'Q2_ingredients', 'Q4_price']:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
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
        
        # Q8 - Hot sauce as categorical
        self.df['Q8_hot_sauce'] = self.df['Q8: How much hot sauce would you add to this food item?'].apply(self.parse_hot_sauce_amount)
        
        # One-hot encode the hot sauce category
        hot_sauce_dummies = pd.get_dummies(self.df['Q8_hot_sauce'], prefix='Q8_hot_sauce')
        self.df = pd.concat([self.df, hot_sauce_dummies], axis=1)
        
        # Encode labels
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['Label'])
        
        # Select features for the model
        numerical_features = ['Q1_complexity', 'Q2_ingredients', 'Q4_price']
        
        # Get setting columns and hot sauce columns
        setting_columns = [col for col in self.df.columns if col.startswith('Q3_setting_')]
        hot_sauce_columns = [col for col in self.df.columns if col.startswith('Q8_hot_sauce_')]
        
        self.feature_names = numerical_features + setting_columns + hot_sauce_columns
        
        # Handle potential NaN values with a simple imputation strategy 
        for col in self.feature_names:
            if self.df[col].dtype in [np.float64, np.int64]:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(0)
        
        print(f"Preprocessed data with {len(self.feature_names)} features")
        return self.df
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets"""
        if self.feature_names is None or len(self.feature_names) == 0:
            self.preprocess_data()
        
        X = self.df[self.feature_names]
        y = self.df['label_encoded']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Split data into {self.X_train.shape[0]} training samples and {self.X_test.shape[0]} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def perform_cross_validation(self, n_folds=5, max_depth=None, min_samples_split=2, random_state=42):
        """Perform k-fold cross-validation"""
        from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
        
        if self.feature_names is None or len(self.feature_names) == 0:
            self.preprocess_data()
        
        X = self.df[self.feature_names]
        y = self.df['label_encoded']
        
        # Initialize the model
        model = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Get predictions from cross-validation for a more detailed report
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Print results
        print(f"Cross-validation results ({n_folds}-fold):")
        print(f"Individual fold accuracies: {[f'{score:.4f}' for score in scores]}")
        print(f"Mean accuracy: {scores.mean():.4f} (std: {scores.std():.4f})")
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y, y_pred, target_names=target_names)
        print("\nClassification Report (Cross-Validation):")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Visualize confusion matrix for cross-validation
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Cross-Validation)')
        plt.colorbar()
        
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the confusion matrix plot
        plt.savefig('cv_confusion_matrix.png')
        plt.close()
        
        # Plot cross-validation results
        self.plot_cross_validation_results(scores, n_folds)
        
        # Train on the entire dataset for feature importance analysis
        self.model = model.fit(X, y)
        
        return scores, self.model
    
    def train_decision_tree(self, max_depth=None, min_samples_split=2, random_state=42):
        """Train a decision tree classifier"""
        if self.X_train is None:
            self.split_data()
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on training data
        train_preds = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_preds)
        
        # Evaluate on testing data
        test_preds = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, test_preds)
        
        print(f"Model trained with {train_acc:.4f} training accuracy and {test_acc:.4f} testing accuracy")
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model with detailed metrics"""
        if self.model is None:
            print("Model not trained yet. Please call train_decision_tree() first.")
            return
        
        y_pred = self.model.predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the confusion matrix plot
        plt.savefig('individual_tree.png')
        plt.close()
        
        return accuracy, report, cm
        
    def plot_cross_validation_results(self, cv_scores, n_folds):
        """Plot cross-validation results"""
        plt.figure(figsize=(10, 6))
        
        # Plot individual fold accuracies
        plt.bar(range(1, n_folds + 1), cv_scores, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Plot mean accuracy line
        mean_accuracy = cv_scores.mean()
        plt.axhline(mean_accuracy, color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean Accuracy: {mean_accuracy:.4f}')
        
        # Customize plot
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title(f'{n_folds}-Fold Cross-Validation Results')
        plt.xticks(range(1, n_folds + 1))
        plt.ylim(min(cv_scores) - 0.05, max(cv_scores) + 0.05)
        
        # Add accuracy values above each bar
        for i, score in enumerate(cv_scores):
            plt.text(i + 1, score + 0.01, f"{score:.4f}", ha='center')
        
        plt.legend()
        plt.tight_layout()
        
        # Save the cross-validation plot
        plt.savefig('individual_tree_results.png')
        plt.close()
        
        print(f"Cross-validation results plot saved to 'cross_validation_results.png'")
        return
    
    def feature_importance(self, top_n=10):
        """Get and visualize feature importance"""
        if self.model is None:
            print("Model not trained yet. Please call train_decision_tree() first.")
            return
        
        # Get feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print feature importance
        print("Feature importance:")
        for i in range(min(top_n, len(self.feature_names))):
            idx = indices[i]
            print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        
        num_features = min(top_n, len(self.feature_names))
        plt.bar(range(num_features), importances[indices[:num_features]])
        plt.xticks(range(num_features), [self.feature_names[i] for i in indices[:num_features]], rotation=90)
        plt.tight_layout()
        
        # Save the feature importance plot
        plt.savefig('feature_importance.png')
        plt.close()
        
        return importances, indices
    
    def save_model(self, model_path='food_survey_decision_tree.joblib'):
        """Save the trained model"""
        if self.model is None:
            print("Model not trained yet. Please call train_decision_tree() first.")
            return
        
        # Create a dictionary with all necessary components for prediction
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path='food_survey_decision_tree.joblib'):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict(self, features_dict):
        """Make a prediction based on survey responses"""
        if self.model is None:
            print("Model not trained yet. Please call train_decision_tree() first or load a trained model.")
            return
        
        # Extract and preprocess features from the input dictionary
        # This simplified version assumes the input dictionary has the exact feature names needed
        # In practice, you'd need more robust preprocessing similar to what's in preprocess_data()
        
        # Initialize a dataframe with zeros for all features
        features = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # Fill in the provided features
        for feature, value in features_dict.items():
            if feature in features.columns:
                features[feature] = value
        
        # Make prediction
        prediction_idx = self.model.predict(features)[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        return prediction
    
    def export_tree_to_python(self, output_file='food_prediction.py'):
        """
        Export the trained decision tree model to a standalone Python file with if-else statements.
        The exported file will not require joblib or sklearn to make predictions.
        
        Parameters:
        -----------
        output_file : str
            Path to the output Python file
        
        Returns:
        --------
        str
            Path to the created Python file
        """
        if self.model is None:
            print("Model not trained yet. Please call train_decision_tree() first.")
            return
        
        tree = self.model.tree_
        feature_names = self.feature_names
        classes = self.label_encoder.classes_
        
        def tree_to_code(tree, feature_names, classes, indent=0):
            """Recursively convert a decision tree to if-else statements."""
            code = []
            
            def recurse(node, depth):
                indent = "    " * depth
                
                if tree.children_left[node] == tree.children_right[node]:  # Leaf node
                    # Get the class with the highest count
                    class_index = tree.value[node].argmax()
                    return [f"{indent}return '{classes[class_index]}'"]
                
                # Feature name and threshold for this node
                feature_index = tree.feature[node]
                threshold = tree.threshold[node]
                feature_name = feature_names[feature_index]
                
                code = []
                # Left branch (feature <= threshold)
                code.append(f"{indent}if features['{feature_name}'] <= {threshold}:")
                code.extend(recurse(tree.children_left[node], depth + 1))
                
                # Right branch (feature > threshold)
                code.append(f"{indent}else:  # {feature_name} > {threshold}")
                code.extend(recurse(tree.children_right[node], depth + 1))
                
                return code
            
            return recurse(0, indent)
        
        # Generate the code for the decision tree
        tree_code = tree_to_code(tree, feature_names, classes)
        
        # Create the full Python file content
        code = [
            "# Food prediction model exported from decision tree",
            "# Generated automatically - do not modify",
            "",
            "import re",
            "",
            "def extract_scale(text):",
            "    \"\"\"Extract scale rating from Q1\"\"\"",
            "    if text is None or text == '':",
            "        return None",
            "    match = re.search(r'(\\d+)', str(text))",
            "    if match:",
            "        num = int(match.group(1))",
            "        if 1 <= num <= 5:",
            "            return num",
            "    return None",
            "",
            "def extract_ingredients(text):",
            "    \"\"\"Extract number of ingredients from Q2\"\"\"",
            "    if text is None or text == '':",
            "        return None",
            "    match = re.search(r'(\\d+)', str(text))",
            "    if match:",
            "        return int(match.group(1))",
            "    return str(text).count(',') + 1",
            "",
            "def text_to_number(text):",
            "    \"\"\"Convert text numbers to actual numbers\"\"\"",
            "    if text is None or text == '':",
            "        return text",
            "    ",
            "    text = str(text).lower()",
            "    number_dict = {",
            "        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,",
            "        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,",
            "        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,",
            "        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50",
            "    }",
            "    ",
            "    compound_pattern = r'(twenty|thirty|forty|fifty)[\\s-](one|two|three|four|five|six|seven|eight|nine)'",
            "    matches = re.finditer(compound_pattern, text)",
            "    for match in matches:",
            "        parts = re.split(r'[\\s-]', match.group(0))",
            "        if len(parts) == 2 and parts[0] in number_dict and parts[1] in number_dict:",
            "            value = number_dict[parts[0]] + number_dict[parts[1]]",
            "            text = text.replace(match.group(0), str(value))",
            "    ",
            "    for word, num in number_dict.items():",
            "        text = re.sub(r'\\b' + word + r'\\b', str(num), text)",
            "    ",
            "    return text",
            "",
            "def extract_price(text):",
            "    \"\"\"Extract price from Q4\"\"\"",
            "    if text is None or text == '':",
            "        return None",
            "    ",
            "    text = text_to_number(str(text).lower())",
            "",
            "    range_match = re.search(r'(\\d+[\\.\\,]?\\d*)\\s*[-–—to]\\s*(\\d+[\\.\\,]?\\d*)', text)",
            "    if range_match:",
            "        try:",
            "            low = float(range_match.group(1).replace(',', '.'))",
            "            high = float(range_match.group(2).replace(',', '.'))",
            "            return (low + high) / 2",
            "        except:",
            "            pass",
            "    ",
            "    match = re.search(r'(\\d+[\\.\\,]?\\d*)', text)",
            "    if match:",
            "        try:",
            "            return float(match.group(1).replace(',', '.'))",
            "        except:",
            "            pass",
            "    ",
            "    return None",
            "",
            "def parse_checkbox_responses(text):",
            "    \"\"\"Parse checkbox responses for Q3\"\"\"",
            "    if text is None or text == '':",
            "        return []",
            "    ",
            "    # Split by common checkbox separators",
            "    options = re.split(r'[,;|]', text)",
            "    ",
            "    # Clean up each option",
            "    options = [opt.strip() for opt in options if opt.strip()]",
            "    ",
            "    return options",
            "",
            "def parse_hot_sauce_amount(text):",
            "    \"\"\"Parse hot sauce amount from Q8\"\"\"",
            "    if text is None or text == '':",
            "        return 'none'",
            "    ",
            "    text = str(text).strip().lower()",
            "    ",
            "    # Direct mapping for the specific values in the dataset",
            "    hot_sauce_mapping = {",
            "        'none': 'none',",
            "        'a little (mild)': 'little',",
            "        'a lot (hot)': 'lot',",
            "        'a moderate amount (medium)': 'medium'",
            "    }",
            "    ",
            "    # Look for exact matches",
            "    for original, category in hot_sauce_mapping.items():",
            "        if original.lower() == text:",
            "            return category",
            "    ",
            "    # Fallback for partial matches",
            "    if 'none' in text:",
            "        return 'none'",
            "    elif 'little' in text or 'mild' in text:",
            "        return 'little'",
            "    elif 'lot' in text or 'hot' in text:",
            "        return 'lot'",
            "    elif 'moderate' in text or 'medium' in text:",
            "        return 'medium'",
            "            ",
            "    return 'other'",
            "",
            "def predict_food(survey_dict):",
            "    \"\"\"",
            "    Predict food type based on survey responses",
            "    ",
            "    Parameters:",
            "    -----------",
            "    survey_dict : dict",
            "        Dictionary with survey responses",
            "        ",
            "    Returns:",
            "    --------",
            "    str",
            "        Predicted food type: 'Pizza', 'Shawarma', or 'Sushi'",
            "    \"\"\"",
            "    # Extract features from the survey dictionary",
            "    features = {}",
            "    ",
            "    # Extract numerical features",
            "    features['Q1_complexity'] = extract_scale(survey_dict.get('Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'))",
            "    features['Q2_ingredients'] = extract_ingredients(survey_dict.get('Q2: How many ingredients would you expect this food item to contain?'))",
            "    features['Q4_price'] = extract_price(survey_dict.get('Q4: How much would you expect to pay for one serving of this food item?'))",
            "    ",
            "    # Fill missing values with defaults",
            "    features['Q1_complexity'] = features['Q1_complexity'] if features['Q1_complexity'] is not None else 3",
            "    features['Q2_ingredients'] = features['Q2_ingredients'] if features['Q2_ingredients'] is not None else 7",
            "    features['Q4_price'] = features['Q4_price'] if features['Q4_price'] is not None else 10",
            "    ",
            "    # Process Q3 - Settings",
            "    q3_text = survey_dict.get('Q3: In what setting would you expect this food to be served? Please check all that apply', '')",
            "    q3_settings = parse_checkbox_responses(q3_text)",
            "    ",
            "    # Initialize all settings to 0",
            "    setting_cols = [",
            "        'Q3_setting_restaurant', 'Q3_setting_fast_food', 'Q3_setting_food_truck',",
            "        'Q3_setting_home_cooked', 'Q3_setting_week_day_lunch', 'Q3_setting_at_a_party',",
            "        'Q3_setting_late_night_snack', 'Q3_setting_street_food', 'Q3_setting_family_dinner'",
            "    ]",
            "    for col in setting_cols:",
            "        features[col] = 0",
            "    ",
            "    # Set specific settings based on responses",
            "    setting_mapping = {",
            "        'restaurant': 'Q3_setting_restaurant',",
            "        'fast food': 'Q3_setting_fast_food',",
            "        'food truck': 'Q3_setting_food_truck',",
            "        'home cooked': 'Q3_setting_home_cooked',",
            "        'week day lunch': 'Q3_setting_week_day_lunch',",
            "        'at a party': 'Q3_setting_at_a_party',",
            "        'late night snack': 'Q3_setting_late_night_snack',",
            "        'street food': 'Q3_setting_street_food',",
            "        'family dinner': 'Q3_setting_family_dinner'",
            "    }",
            "    ",
            "    for setting in q3_settings:",
            "        setting_lower = setting.lower()",
            "        for key, col in setting_mapping.items():",
            "            if key in setting_lower:",
            "                features[col] = 1",
            "                break",
            "    ",
            "    # Process Q8 - Hot sauce",
            "    q8_text = survey_dict.get('Q8: How much hot sauce would you add to this food item?', '')",
            "    hot_sauce_type = parse_hot_sauce_amount(q8_text)",
            "    ",
            "    # Initialize all hot sauce types to 0",
            "    hot_sauce_cols = ['Q8_hot_sauce_none', 'Q8_hot_sauce_little', 'Q8_hot_sauce_medium', 'Q8_hot_sauce_lot', 'Q8_hot_sauce_other']",
            "    for col in hot_sauce_cols:",
            "        features[col] = 0",
            "    ",
            "    # Set specific hot sauce type based on response",
            "    col_name = f'Q8_hot_sauce_{hot_sauce_type}'",
            "    if col_name in hot_sauce_cols:",
            "        features[col_name] = 1",
            "    else:",
            "        features['Q8_hot_sauce_other'] = 1",
            "    ",
            "    # Now run the decision tree logic",
        ]
        
        # Add the decision tree if-else statements
        code.extend(tree_code)
        
        # Add example usage
        code.extend([
            "",
            "# Example usage",
            "if __name__ == '__main__':",
            "    example = {",
            "        'id': '716549',", 
            "        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': '3',", 
            "        'Q2: How many ingredients would you expect this food item to contain?': '6',", 
            "        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Week day lunch,At a party,Late night snack',", 
            "        'Q4: How much would you expect to pay for one serving of this food item?': '5',", 
            "        'Q5: What movie do you think of when thinking of this food item?': 'Cloudy with a Chance of Meatballs',", 
            "        'Q6: What drink would you pair with this food item?': 'Coke\\xa0',", 
            "        'Q7: When you think about this food item, who does it remind you of?': 'Friends',", 
            "        'Q8: How much hot sauce would you add to this food item?': 'A little (mild)',", 
            "        'Label': 'Pizza'",
            "    }",
            "    prediction = predict_food(example)",
            "    print(f'Predicted food: {prediction}')",
            "    # You can test with other examples or without the 'Label' key to see predictions"
        ])
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(code))
        
        print(f"Decision tree exported to {output_file}")
        return output_file


def main():
    # Example usage
    csv_path = '../cleaned_data_combined_modified.csv'
    
    # Initialize the data loader
    loader = FoodSurveyDataLoader(csv_path)
    
    # Load and preprocess the data
    loader.load_data()
    loader.preprocess_data()
    
    # Perform cross-validation (k-fold)
    print("\n=== Cross-Validation Evaluation ===")
    scores, model = loader.perform_cross_validation(n_folds=5, max_depth=5)
    
    print("\n=== Feature Importance Analysis ===")
    loader.feature_importance(top_n=10)
    

    print("\n=== Train/Test Split Evaluation ===")
    loader.split_data(test_size=0.2)
    loader.train_decision_tree(max_depth=10)
    loader.export_tree_to_python('food_prediction.py')
    loader.evaluate_model()
    
    loader.save_model()
    
    print("\n=== Example Prediction ===")
    # loader.load_model()
    
    example_features = {
        'Q1_complexity': 3,
        'Q2_ingredients': 8,
        'Q4_price': 12.99,
        'Q3_setting_restaurant': 1,
        'Q8_hot_sauce_lot': 1
    }
    
    prediction = loader.predict(example_features)
    print(f"Predicted food item: {prediction}")


if __name__ == "__main__":
    main()