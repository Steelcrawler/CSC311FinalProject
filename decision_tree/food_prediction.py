# Food prediction model exported from decision tree
# Generated automatically - do not modify

import re

def extract_scale(text):
    """Extract scale rating from Q1"""
    if text is None or text == '':
        return None
    match = re.search(r'(\d+)', str(text))
    if match:
        num = int(match.group(1))
        if 1 <= num <= 5:
            return num
    return None

def extract_ingredients(text):
    """Extract number of ingredients from Q2"""
    if text is None or text == '':
        return None
    match = re.search(r'(\d+)', str(text))
    if match:
        return int(match.group(1))
    return str(text).count(',') + 1

def text_to_number(text):
    """Convert text numbers to actual numbers"""
    if text is None or text == '':
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

def extract_price(text):
    """Extract price from Q4"""
    if text is None or text == '':
        return None
    
    text = text_to_number(str(text).lower())

    range_match = re.search(r'(\d+[\.\,]?\d*)\s*[-–—to]\s*(\d+[\.\,]?\d*)', text)
    if range_match:
        try:
            low = float(range_match.group(1).replace(',', '.'))
            high = float(range_match.group(2).replace(',', '.'))
            return (low + high) / 2
        except:
            pass
    
    match = re.search(r'(\d+[\.\,]?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', '.'))
        except:
            pass
    
    return None

def parse_checkbox_responses(text):
    """Parse checkbox responses for Q3"""
    if text is None or text == '':
        return []
    
    # Split by common checkbox separators
    options = re.split(r'[,;|]', text)
    
    # Clean up each option
    options = [opt.strip() for opt in options if opt.strip()]
    
    return options

def parse_hot_sauce_amount(text):
    """Parse hot sauce amount from Q8"""
    if text is None or text == '':
        return 'none'
    
    text = str(text).strip().lower()
    
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

def predict_food(survey_dict):
    """
    Predict food type based on survey responses
    
    Parameters:
    -----------
    survey_dict : dict
        Dictionary with survey responses
        
    Returns:
    --------
    str
        Predicted food type: 'Pizza', 'Shawarma', or 'Sushi'
    """
    # Extract features from the survey dictionary
    features = {}
    
    # Extract numerical features
    features['Q1_complexity'] = extract_scale(survey_dict.get('Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'))
    features['Q2_ingredients'] = extract_ingredients(survey_dict.get('Q2: How many ingredients would you expect this food item to contain?'))
    features['Q4_price'] = extract_price(survey_dict.get('Q4: How much would you expect to pay for one serving of this food item?'))
    
    # Fill missing values with defaults
    features['Q1_complexity'] = features['Q1_complexity'] if features['Q1_complexity'] is not None else 3
    features['Q2_ingredients'] = features['Q2_ingredients'] if features['Q2_ingredients'] is not None else 7
    features['Q4_price'] = features['Q4_price'] if features['Q4_price'] is not None else 10
    
    # Process Q3 - Settings
    q3_text = survey_dict.get('Q3: In what setting would you expect this food to be served? Please check all that apply', '')
    q3_settings = parse_checkbox_responses(q3_text)
    
    # Initialize all settings to 0
    setting_cols = [
        'Q3_setting_restaurant', 'Q3_setting_fast_food', 'Q3_setting_food_truck',
        'Q3_setting_home_cooked', 'Q3_setting_week_day_lunch', 'Q3_setting_at_a_party',
        'Q3_setting_late_night_snack', 'Q3_setting_street_food', 'Q3_setting_family_dinner'
    ]
    for col in setting_cols:
        features[col] = 0
    
    # Set specific settings based on responses
    setting_mapping = {
        'restaurant': 'Q3_setting_restaurant',
        'fast food': 'Q3_setting_fast_food',
        'food truck': 'Q3_setting_food_truck',
        'home cooked': 'Q3_setting_home_cooked',
        'week day lunch': 'Q3_setting_week_day_lunch',
        'at a party': 'Q3_setting_at_a_party',
        'late night snack': 'Q3_setting_late_night_snack',
        'street food': 'Q3_setting_street_food',
        'family dinner': 'Q3_setting_family_dinner'
    }
    
    for setting in q3_settings:
        setting_lower = setting.lower()
        for key, col in setting_mapping.items():
            if key in setting_lower:
                features[col] = 1
                break
    
    # Process Q8 - Hot sauce
    q8_text = survey_dict.get('Q8: How much hot sauce would you add to this food item?', '')
    hot_sauce_type = parse_hot_sauce_amount(q8_text)
    
    # Initialize all hot sauce types to 0
    hot_sauce_cols = ['Q8_hot_sauce_none', 'Q8_hot_sauce_little', 'Q8_hot_sauce_medium', 'Q8_hot_sauce_lot', 'Q8_hot_sauce_other']
    for col in hot_sauce_cols:
        features[col] = 0
    
    # Set specific hot sauce type based on response
    col_name = f'Q8_hot_sauce_{hot_sauce_type}'
    if col_name in hot_sauce_cols:
        features[col_name] = 1
    else:
        features['Q8_hot_sauce_other'] = 1
    
    # Now run the decision tree logic
if features['Q3_setting_at_a_party'] <= 0.5:
    if features['Q8_hot_sauce_none'] <= 0.5:
        if features['Q2_ingredients'] <= 5.5:
            if features['Q4_price'] <= 16.75:
                if features['Q3_setting_week_day_lunch'] <= 0.5:
                    if features['Q4_price'] <= 9.5:
                        if features['Q3_setting_week_day_dinner'] <= 0.5:
                            return 'Sushi'
                        else:  # Q3_setting_week_day_dinner > 0.5
                            if features['Q4_price'] <= 6.5:
                                if features['Q8_hot_sauce_little'] <= 0.5:
                                    return 'Sushi'
                                else:  # Q8_hot_sauce_little > 0.5
                                    return 'Shawarma'
                            else:  # Q4_price > 6.5
                                return 'Shawarma'
                    else:  # Q4_price > 9.5
                        if features['Q2_ingredients'] <= 3.5:
                            if features['Q1_complexity'] <= 4.5:
                                return 'Shawarma'
                            else:  # Q1_complexity > 4.5
                                return 'Sushi'
                        else:  # Q2_ingredients > 3.5
                            if features['Q3_setting_weekend_dinner'] <= 0.5:
                                if features['Q3_setting_week_day_dinner'] <= 0.5:
                                    return 'Sushi'
                                else:  # Q3_setting_week_day_dinner > 0.5
                                    if features['Q4_price'] <= 11.0:
                                        return 'Shawarma'
                                    else:  # Q4_price > 11.0
                                        return 'Sushi'
                            else:  # Q3_setting_weekend_dinner > 0.5
                                return 'Sushi'
                else:  # Q3_setting_week_day_lunch > 0.5
                    if features['Q4_price'] <= 4.5:
                        return 'Sushi'
                    else:  # Q4_price > 4.5
                        if features['Q1_complexity'] <= 3.5:
                            if features['Q4_price'] <= 7.5:
                                if features['Q1_complexity'] <= 2.5:
                                    if features['Q3_setting_late_night_snack'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q3_setting_late_night_snack > 0.5
                                        return 'Sushi'
                                else:  # Q1_complexity > 2.5
                                    return 'Shawarma'
                            else:  # Q4_price > 7.5
                                if features['Q4_price'] <= 14.0:
                                    if features['Q8_hot_sauce_medium'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q8_hot_sauce_medium > 0.5
                                        return 'Shawarma'
                                else:  # Q4_price > 14.0
                                    if features['Q8_hot_sauce_lot'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q8_hot_sauce_lot > 0.5
                                        return 'Pizza'
                        else:  # Q1_complexity > 3.5
                            if features['Q3_setting_weekend_dinner'] <= 0.5:
                                if features['Q4_price'] <= 13.5:
                                    return 'Shawarma'
                                else:  # Q4_price > 13.5
                                    if features['Q3_setting_week_day_dinner'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q3_setting_week_day_dinner > 0.5
                                        return 'Sushi'
                            else:  # Q3_setting_weekend_dinner > 0.5
                                if features['Q4_price'] <= 11.5:
                                    if features['Q2_ingredients'] <= 1.5:
                                        return 'Shawarma'
                                    else:  # Q2_ingredients > 1.5
                                        return 'Sushi'
                                else:  # Q4_price > 11.5
                                    return 'Shawarma'
            else:  # Q4_price > 16.75
                if features['Q3_setting_weekend_dinner'] <= 0.5:
                    if features['Q3_setting_week_day_lunch'] <= 0.5:
                        return 'Shawarma'
                    else:  # Q3_setting_week_day_lunch > 0.5
                        if features['Q2_ingredients'] <= 4.0:
                            return 'Shawarma'
                        else:  # Q2_ingredients > 4.0
                            return 'Sushi'
                else:  # Q3_setting_weekend_dinner > 0.5
                    return 'Sushi'
        else:  # Q2_ingredients > 5.5
            if features['Q4_price'] <= 22.5:
                if features['Q4_price'] <= 4.0:
                    if features['Q4_price'] <= 2.75:
                        if features['Q2_ingredients'] <= 6.5:
                            return 'Sushi'
                        else:  # Q2_ingredients > 6.5
                            return 'Shawarma'
                    else:  # Q4_price > 2.75
                        return 'Pizza'
                else:  # Q4_price > 4.0
                    if features['Q4_price'] <= 19.0:
                        if features['Q4_price'] <= 14.5:
                            if features['Q2_ingredients'] <= 9.5:
                                if features['Q2_ingredients'] <= 6.5:
                                    if features['Q8_hot_sauce_little'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q8_hot_sauce_little > 0.5
                                        return 'Shawarma'
                                else:  # Q2_ingredients > 6.5
                                    return 'Shawarma'
                            else:  # Q2_ingredients > 9.5
                                if features['Q3_setting_week_day_lunch'] <= 0.5:
                                    if features['Q3_setting_weekend_dinner'] <= 0.5:
                                        return 'Pizza'
                                    else:  # Q3_setting_weekend_dinner > 0.5
                                        return 'Shawarma'
                                else:  # Q3_setting_week_day_lunch > 0.5
                                    if features['Q3_setting_week_day_dinner'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q3_setting_week_day_dinner > 0.5
                                        return 'Shawarma'
                        else:  # Q4_price > 14.5
                            if features['Q2_ingredients'] <= 9.0:
                                if features['Q8_hot_sauce_little'] <= 0.5:
                                    if features['Q1_complexity'] <= 4.5:
                                        return 'Shawarma'
                                    else:  # Q1_complexity > 4.5
                                        return 'Shawarma'
                                else:  # Q8_hot_sauce_little > 0.5
                                    if features['Q1_complexity'] <= 2.5:
                                        return 'Shawarma'
                                    else:  # Q1_complexity > 2.5
                                        return 'Sushi'
                            else:  # Q2_ingredients > 9.0
                                return 'Shawarma'
                    else:  # Q4_price > 19.0
                        if features['Q1_complexity'] <= 4.5:
                            if features['Q1_complexity'] <= 3.5:
                                return 'Shawarma'
                            else:  # Q1_complexity > 3.5
                                if features['Q2_ingredients'] <= 9.0:
                                    return 'Sushi'
                                else:  # Q2_ingredients > 9.0
                                    return 'Shawarma'
                        else:  # Q1_complexity > 4.5
                            return 'Sushi'
            else:  # Q4_price > 22.5
                if features['Q3_setting_late_night_snack'] <= 0.5:
                    return 'Sushi'
                else:  # Q3_setting_late_night_snack > 0.5
                    return 'Pizza'
    else:  # Q8_hot_sauce_none > 0.5
        if features['Q3_setting_weekend_dinner'] <= 0.5:
            if features['Q3_setting_week_day_lunch'] <= 0.5:
                if features['Q3_setting_late_night_snack'] <= 0.5:
                    if features['Q2_ingredients'] <= 4.5:
                        return 'Sushi'
                    else:  # Q2_ingredients > 4.5
                        if features['Q4_price'] <= 22.5:
                            return 'Sushi'
                        else:  # Q4_price > 22.5
                            return 'Pizza'
                else:  # Q3_setting_late_night_snack > 0.5
                    return 'Shawarma'
            else:  # Q3_setting_week_day_lunch > 0.5
                if features['Q4_price'] <= 5.494999885559082:
                    if features['Q1_complexity'] <= 3.5:
                        if features['Q1_complexity'] <= 2.5:
                            return 'Sushi'
                        else:  # Q1_complexity > 2.5
                            return 'Pizza'
                    else:  # Q1_complexity > 3.5
                        if features['Q4_price'] <= 2.5:
                            return 'Shawarma'
                        else:  # Q4_price > 2.5
                            return 'Sushi'
                else:  # Q4_price > 5.494999885559082
                    if features['Q2_ingredients'] <= 1.5:
                        return 'Sushi'
                    else:  # Q2_ingredients > 1.5
                        if features['Q2_ingredients'] <= 16.5:
                            if features['Q2_ingredients'] <= 5.5:
                                if features['Q4_price'] <= 11.994999885559082:
                                    if features['Q2_ingredients'] <= 4.5:
                                        return 'Shawarma'
                                    else:  # Q2_ingredients > 4.5
                                        return 'Shawarma'
                                else:  # Q4_price > 11.994999885559082
                                    if features['Q4_price'] <= 13.5:
                                        return 'Sushi'
                                    else:  # Q4_price > 13.5
                                        return 'Shawarma'
                            else:  # Q2_ingredients > 5.5
                                if features['Q4_price'] <= 14.75:
                                    return 'Shawarma'
                                else:  # Q4_price > 14.75
                                    if features['Q4_price'] <= 15.5:
                                        return 'Sushi'
                                    else:  # Q4_price > 15.5
                                        return 'Shawarma'
                        else:  # Q2_ingredients > 16.5
                            return 'Sushi'
        else:  # Q3_setting_weekend_dinner > 0.5
            if features['Q3_setting_late_night_snack'] <= 0.5:
                if features['Q2_ingredients'] <= 13.0:
                    if features['Q4_price'] <= 12.5:
                        if features['Q3_setting_weekend_lunch'] <= 0.5:
                            return 'Sushi'
                        else:  # Q3_setting_weekend_lunch > 0.5
                            if features['Q4_price'] <= 7.5:
                                return 'Sushi'
                            else:  # Q4_price > 7.5
                                if features['Q2_ingredients'] <= 3.5:
                                    return 'Sushi'
                                else:  # Q2_ingredients > 3.5
                                    if features['Q1_complexity'] <= 2.5:
                                        return 'Sushi'
                                    else:  # Q1_complexity > 2.5
                                        return 'Sushi'
                    else:  # Q4_price > 12.5
                        if features['Q1_complexity'] <= 3.5:
                            if features['Q1_complexity'] <= 2.5:
                                return 'Sushi'
                            else:  # Q1_complexity > 2.5
                                if features['Q2_ingredients'] <= 3.5:
                                    return 'Sushi'
                                else:  # Q2_ingredients > 3.5
                                    if features['Q4_price'] <= 18.75:
                                        return 'Sushi'
                                    else:  # Q4_price > 18.75
                                        return 'Sushi'
                        else:  # Q1_complexity > 3.5
                            return 'Sushi'
                else:  # Q2_ingredients > 13.0
                    return 'Shawarma'
            else:  # Q3_setting_late_night_snack > 0.5
                if features['Q3_setting_weekend_lunch'] <= 0.5:
                    return 'Sushi'
                else:  # Q3_setting_weekend_lunch > 0.5
                    if features['Q4_price'] <= 18.75:
                        if features['Q4_price'] <= 16.25:
                            if features['Q2_ingredients'] <= 3.5:
                                if features['Q4_price'] <= 13.5:
                                    return 'Sushi'
                                else:  # Q4_price > 13.5
                                    if features['Q1_complexity'] <= 3.5:
                                        return 'Pizza'
                                    else:  # Q1_complexity > 3.5
                                        return 'Sushi'
                            else:  # Q2_ingredients > 3.5
                                if features['Q4_price'] <= 12.75:
                                    if features['Q4_price'] <= 6.75:
                                        return 'Sushi'
                                    else:  # Q4_price > 6.75
                                        return 'Shawarma'
                                else:  # Q4_price > 12.75
                                    return 'Sushi'
                        else:  # Q4_price > 16.25
                            return 'Shawarma'
                    else:  # Q4_price > 18.75
                        return 'Sushi'
else:  # Q3_setting_at_a_party > 0.5
    if features['Q4_price'] <= 6.25:
        if features['Q1_complexity'] <= 4.5:
            if features['Q2_ingredients'] <= 2.5:
                if features['Q2_ingredients'] <= 1.5:
                    if features['Q4_price'] <= 5.5:
                        if features['Q3_setting_week_day_lunch'] <= 0.5:
                            return 'Sushi'
                        else:  # Q3_setting_week_day_lunch > 0.5
                            return 'Pizza'
                    else:  # Q4_price > 5.5
                        return 'Shawarma'
                else:  # Q2_ingredients > 1.5
                    return 'Sushi'
            else:  # Q2_ingredients > 2.5
                if features['Q4_price'] <= 1.7450000047683716:
                    if features['Q8_hot_sauce_lot'] <= 0.5:
                        if features['Q4_price'] <= 1.2450000047683716:
                            if features['Q1_complexity'] <= 3.5:
                                if features['Q2_ingredients'] <= 3.5:
                                    if features['Q1_complexity'] <= 2.5:
                                        return 'Pizza'
                                    else:  # Q1_complexity > 2.5
                                        return 'Sushi'
                                else:  # Q2_ingredients > 3.5
                                    return 'Pizza'
                            else:  # Q1_complexity > 3.5
                                if features['Q3_setting_weekend_dinner'] <= 0.5:
                                    return 'Pizza'
                                else:  # Q3_setting_weekend_dinner > 0.5
                                    return 'Sushi'
                        else:  # Q4_price > 1.2450000047683716
                            return 'Sushi'
                    else:  # Q8_hot_sauce_lot > 0.5
                        return 'Sushi'
                else:  # Q4_price > 1.7450000047683716
                    if features['Q3_setting_late_night_snack'] <= 0.5:
                        if features['Q2_ingredients'] <= 3.5:
                            if features['Q8_hot_sauce_little'] <= 0.5:
                                if features['Q3_setting_week_day_dinner'] <= 0.5:
                                    if features['Q1_complexity'] <= 2.5:
                                        return 'Pizza'
                                    else:  # Q1_complexity > 2.5
                                        return 'Pizza'
                                else:  # Q3_setting_week_day_dinner > 0.5
                                    if features['Q8_hot_sauce_lot'] <= 0.5:
                                        return 'Sushi'
                                    else:  # Q8_hot_sauce_lot > 0.5
                                        return 'Shawarma'
                            else:  # Q8_hot_sauce_little > 0.5
                                return 'Pizza'
                        else:  # Q2_ingredients > 3.5
                            if features['Q4_price'] <= 4.994999885559082:
                                return 'Pizza'
                            else:  # Q4_price > 4.994999885559082
                                if features['Q8_hot_sauce_little'] <= 0.5:
                                    if features['Q3_setting_weekend_dinner'] <= 0.5:
                                        return 'Pizza'
                                    else:  # Q3_setting_weekend_dinner > 0.5
                                        return 'Pizza'
                                else:  # Q8_hot_sauce_little > 0.5
                                    if features['Q4_price'] <= 5.5:
                                        return 'Sushi'
                                    else:  # Q4_price > 5.5
                                        return 'Pizza'
                    else:  # Q3_setting_late_night_snack > 0.5
                        if features['Q1_complexity'] <= 3.5:
                            if features['Q8_hot_sauce_lot'] <= 0.5:
                                if features['Q2_ingredients'] <= 3.5:
                                    if features['Q4_price'] <= 4.5:
                                        return 'Pizza'
                                    else:  # Q4_price > 4.5
                                        return 'Pizza'
                                else:  # Q2_ingredients > 3.5
                                    if features['Q8_hot_sauce_little'] <= 0.5:
                                        return 'Pizza'
                                    else:  # Q8_hot_sauce_little > 0.5
                                        return 'Pizza'
                            else:  # Q8_hot_sauce_lot > 0.5
                                if features['Q2_ingredients'] <= 4.5:
                                    return 'Sushi'
                                else:  # Q2_ingredients > 4.5
                                    return 'Pizza'
                        else:  # Q1_complexity > 3.5
                            if features['Q2_ingredients'] <= 5.5:
                                if features['Q4_price'] <= 5.75:
                                    if features['Q4_price'] <= 4.200000047683716:
                                        return 'Pizza'
                                    else:  # Q4_price > 4.200000047683716
                                        return 'Pizza'
                                else:  # Q4_price > 5.75
                                    return 'Sushi'
                            else:  # Q2_ingredients > 5.5
                                return 'Pizza'
        else:  # Q1_complexity > 4.5
            if features['Q3_setting_late_night_snack'] <= 0.5:
                return 'Sushi'
            else:  # Q3_setting_late_night_snack > 0.5
                if features['Q3_setting_weekend_lunch'] <= 0.5:
                    return 'Pizza'
                else:  # Q3_setting_weekend_lunch > 0.5
                    if features['Q4_price'] <= 3.5:
                        return 'Sushi'
                    else:  # Q4_price > 3.5
                        if features['Q2_ingredients'] <= 2.5:
                            return 'Sushi'
                        else:  # Q2_ingredients > 2.5
                            return 'Shawarma'
    else:  # Q4_price > 6.25
        if features['Q1_complexity'] <= 3.5:
            if features['Q2_ingredients'] <= 2.5:
                if features['Q4_price'] <= 9.0:
                    return 'Pizza'
                else:  # Q4_price > 9.0
                    return 'Sushi'
            else:  # Q2_ingredients > 2.5
                if features['Q4_price'] <= 14.5:
                    if features['Q8_hot_sauce_none'] <= 0.5:
                        if features['Q3_setting_weekend_lunch'] <= 0.5:
                            if features['Q1_complexity'] <= 2.5:
                                if features['Q3_setting_week_day_dinner'] <= 0.5:
                                    return 'Pizza'
                                else:  # Q3_setting_week_day_dinner > 0.5
                                    if features['Q2_ingredients'] <= 5.0:
                                        return 'Shawarma'
                                    else:  # Q2_ingredients > 5.0
                                        return 'Pizza'
                            else:  # Q1_complexity > 2.5
                                if features['Q3_setting_weekend_dinner'] <= 0.5:
                                    if features['Q2_ingredients'] <= 7.0:
                                        return 'Pizza'
                                    else:  # Q2_ingredients > 7.0
                                        return 'Pizza'
                                else:  # Q3_setting_weekend_dinner > 0.5
                                    if features['Q4_price'] <= 8.25:
                                        return 'Pizza'
                                    else:  # Q4_price > 8.25
                                        return 'Sushi'
                        else:  # Q3_setting_weekend_lunch > 0.5
                            if features['Q3_setting_late_night_snack'] <= 0.5:
                                if features['Q2_ingredients'] <= 4.5:
                                    if features['Q4_price'] <= 11.0:
                                        return 'Shawarma'
                                    else:  # Q4_price > 11.0
                                        return 'Pizza'
                                else:  # Q2_ingredients > 4.5
                                    if features['Q4_price'] <= 7.5:
                                        return 'Shawarma'
                                    else:  # Q4_price > 7.5
                                        return 'Shawarma'
                            else:  # Q3_setting_late_night_snack > 0.5
                                if features['Q2_ingredients'] <= 6.5:
                                    if features['Q4_price'] <= 12.25:
                                        return 'Pizza'
                                    else:  # Q4_price > 12.25
                                        return 'Shawarma'
                                else:  # Q2_ingredients > 6.5
                                    if features['Q8_hot_sauce_little'] <= 0.5:
                                        return 'Shawarma'
                                    else:  # Q8_hot_sauce_little > 0.5
                                        return 'Pizza'
                    else:  # Q8_hot_sauce_none > 0.5
                        if features['Q4_price'] <= 8.25:
                            if features['Q2_ingredients'] <= 5.5:
                                if features['Q2_ingredients'] <= 3.5:
                                    return 'Pizza'
                                else:  # Q2_ingredients > 3.5
                                    if features['Q1_complexity'] <= 1.5:
                                        return 'Sushi'
                                    else:  # Q1_complexity > 1.5
                                        return 'Pizza'
                            else:  # Q2_ingredients > 5.5
                                return 'Pizza'
                        else:  # Q4_price > 8.25
                            if features['Q2_ingredients'] <= 4.5:
                                if features['Q4_price'] <= 11.75:
                                    if features['Q2_ingredients'] <= 3.5:
                                        return 'Sushi'
                                    else:  # Q2_ingredients > 3.5
                                        return 'Pizza'
                                else:  # Q4_price > 11.75
                                    if features['Q4_price'] <= 13.0:
                                        return 'Pizza'
                                    else:  # Q4_price > 13.0
                                        return 'Sushi'
                            else:  # Q2_ingredients > 4.5
                                if features['Q3_setting_weekend_lunch'] <= 0.5:
                                    if features['Q2_ingredients'] <= 7.0:
                                        return 'Sushi'
                                    else:  # Q2_ingredients > 7.0
                                        return 'Pizza'
                                else:  # Q3_setting_weekend_lunch > 0.5
                                    if features['Q2_ingredients'] <= 9.0:
                                        return 'Pizza'
                                    else:  # Q2_ingredients > 9.0
                                        return 'Sushi'
                else:  # Q4_price > 14.5
                    if features['Q1_complexity'] <= 1.5:
                        if features['Q8_hot_sauce_lot'] <= 0.5:
                            if features['Q2_ingredients'] <= 3.5:
                                if features['Q3_setting_late_night_snack'] <= 0.5:
                                    return 'Sushi'
                                else:  # Q3_setting_late_night_snack > 0.5
                                    return 'Shawarma'
                            else:  # Q2_ingredients > 3.5
                                return 'Sushi'
                        else:  # Q8_hot_sauce_lot > 0.5
                            return 'Pizza'
                    else:  # Q1_complexity > 1.5
                        if features['Q3_setting_week_day_lunch'] <= 0.5:
                            if features['Q8_hot_sauce_lot'] <= 0.5:
                                if features['Q8_hot_sauce_none'] <= 0.5:
                                    if features['Q2_ingredients'] <= 3.5:
                                        return 'Pizza'
                                    else:  # Q2_ingredients > 3.5
                                        return 'Pizza'
                                else:  # Q8_hot_sauce_none > 0.5
                                    if features['Q3_setting_late_night_snack'] <= 0.5:
                                        return 'Sushi'
                                    else:  # Q3_setting_late_night_snack > 0.5
                                        return 'Pizza'
                            else:  # Q8_hot_sauce_lot > 0.5
                                return 'Shawarma'
                        else:  # Q3_setting_week_day_lunch > 0.5
                            if features['Q2_ingredients'] <= 12.5:
                                if features['Q4_price'] <= 37.5:
                                    if features['Q4_price'] <= 15.5:
                                        return 'Pizza'
                                    else:  # Q4_price > 15.5
                                        return 'Pizza'
                                else:  # Q4_price > 37.5
                                    return 'Sushi'
                            else:  # Q2_ingredients > 12.5
                                return 'Shawarma'
        else:  # Q1_complexity > 3.5
            if features['Q3_setting_late_night_snack'] <= 0.5:
                if features['Q3_setting_week_day_lunch'] <= 0.5:
                    if features['Q3_setting_weekend_lunch'] <= 0.5:
                        if features['Q8_hot_sauce_little'] <= 0.5:
                            if features['Q3_setting_week_day_dinner'] <= 0.5:
                                if features['Q1_complexity'] <= 4.5:
                                    if features['Q4_price'] <= 11.5:
                                        return 'Pizza'
                                    else:  # Q4_price > 11.5
                                        return 'Sushi'
                                else:  # Q1_complexity > 4.5
                                    return 'Sushi'
                            else:  # Q3_setting_week_day_dinner > 0.5
                                return 'Sushi'
                        else:  # Q8_hot_sauce_little > 0.5
                            if features['Q4_price'] <= 13.5:
                                return 'Sushi'
                            else:  # Q4_price > 13.5
                                if features['Q4_price'] <= 17.5:
                                    return 'Pizza'
                                else:  # Q4_price > 17.5
                                    return 'Sushi'
                    else:  # Q3_setting_weekend_lunch > 0.5
                        if features['Q2_ingredients'] <= 5.5:
                            if features['Q4_price'] <= 21.25:
                                if features['Q3_setting_weekend_dinner'] <= 0.5:
                                    return 'Pizza'
                                else:  # Q3_setting_weekend_dinner > 0.5
                                    if features['Q8_hot_sauce_lot'] <= 0.5:
                                        return 'Sushi'
                                    else:  # Q8_hot_sauce_lot > 0.5
                                        return 'Pizza'
                            else:  # Q4_price > 21.25
                                return 'Pizza'
                        else:  # Q2_ingredients > 5.5
                            if features['Q2_ingredients'] <= 9.5:
                                return 'Pizza'
                            else:  # Q2_ingredients > 9.5
                                return 'Sushi'
                else:  # Q3_setting_week_day_lunch > 0.5
                    if features['Q8_hot_sauce_none'] <= 0.5:
                        if features['Q8_hot_sauce_little'] <= 0.5:
                            if features['Q4_price'] <= 9.0:
                                if features['Q2_ingredients'] <= 5.5:
                                    return 'Pizza'
                                else:  # Q2_ingredients > 5.5
                                    return 'Sushi'
                            else:  # Q4_price > 9.0
                                return 'Shawarma'
                        else:  # Q8_hot_sauce_little > 0.5
                            if features['Q1_complexity'] <= 4.5:
                                if features['Q3_setting_week_day_dinner'] <= 0.5:
                                    return 'Sushi'
                                else:  # Q3_setting_week_day_dinner > 0.5
                                    if features['Q2_ingredients'] <= 13.0:
                                        return 'Shawarma'
                                    else:  # Q2_ingredients > 13.0
                                        return 'Pizza'
                            else:  # Q1_complexity > 4.5
                                return 'Sushi'
                    else:  # Q8_hot_sauce_none > 0.5
                        if features['Q3_setting_weekend_lunch'] <= 0.5:
                            if features['Q3_setting_week_day_dinner'] <= 0.5:
                                if features['Q4_price'] <= 12.5:
                                    return 'Pizza'
                                else:  # Q4_price > 12.5
                                    return 'Sushi'
                            else:  # Q3_setting_week_day_dinner > 0.5
                                return 'Pizza'
                        else:  # Q3_setting_weekend_lunch > 0.5
                            if features['Q2_ingredients'] <= 1.5:
                                return 'Shawarma'
                            else:  # Q2_ingredients > 1.5
                                if features['Q3_setting_weekend_dinner'] <= 0.5:
                                    if features['Q1_complexity'] <= 4.5:
                                        return 'Sushi'
                                    else:  # Q1_complexity > 4.5
                                        return 'Shawarma'
                                else:  # Q3_setting_weekend_dinner > 0.5
                                    if features['Q2_ingredients'] <= 7.0:
                                        return 'Sushi'
                                    else:  # Q2_ingredients > 7.0
                                        return 'Sushi'
            else:  # Q3_setting_late_night_snack > 0.5
                if features['Q8_hot_sauce_medium'] <= 0.5:
                    if features['Q2_ingredients'] <= 4.5:
                        if features['Q8_hot_sauce_none'] <= 0.5:
                            if features['Q3_setting_weekend_lunch'] <= 0.5:
                                if features['Q1_complexity'] <= 4.5:
                                    return 'Pizza'
                                else:  # Q1_complexity > 4.5
                                    return 'Sushi'
                            else:  # Q3_setting_weekend_lunch > 0.5
                                if features['Q8_hot_sauce_lot'] <= 0.5:
                                    if features['Q4_price'] <= 17.5:
                                        return 'Shawarma'
                                    else:  # Q4_price > 17.5
                                        return 'Sushi'
                                else:  # Q8_hot_sauce_lot > 0.5
                                    return 'Sushi'
                        else:  # Q8_hot_sauce_none > 0.5
                            if features['Q3_setting_week_day_dinner'] <= 0.5:
                                if features['Q1_complexity'] <= 4.5:
                                    return 'Sushi'
                                else:  # Q1_complexity > 4.5
                                    return 'Pizza'
                            else:  # Q3_setting_week_day_dinner > 0.5
                                return 'Sushi'
                    else:  # Q2_ingredients > 4.5
                        if features['Q4_price'] <= 11.0:
                            if features['Q1_complexity'] <= 4.5:
                                if features['Q3_setting_weekend_dinner'] <= 0.5:
                                    if features['Q4_price'] <= 9.5:
                                        return 'Shawarma'
                                    else:  # Q4_price > 9.5
                                        return 'Sushi'
                                else:  # Q3_setting_weekend_dinner > 0.5
                                    if features['Q4_price'] <= 7.5:
                                        return 'Pizza'
                                    else:  # Q4_price > 7.5
                                        return 'Pizza'
                            else:  # Q1_complexity > 4.5
                                if features['Q2_ingredients'] <= 7.5:
                                    if features['Q4_price'] <= 8.5:
                                        return 'Shawarma'
                                    else:  # Q4_price > 8.5
                                        return 'Sushi'
                                else:  # Q2_ingredients > 7.5
                                    return 'Shawarma'
                        else:  # Q4_price > 11.0
                            if features['Q4_price'] <= 16.5:
                                return 'Pizza'
                            else:  # Q4_price > 16.5
                                if features['Q3_setting_week_day_dinner'] <= 0.5:
                                    return 'Pizza'
                                else:  # Q3_setting_week_day_dinner > 0.5
                                    if features['Q1_complexity'] <= 4.5:
                                        return 'Sushi'
                                    else:  # Q1_complexity > 4.5
                                        return 'Pizza'
                else:  # Q8_hot_sauce_medium > 0.5
                    if features['Q4_price'] <= 13.994999885559082:
                        if features['Q2_ingredients'] <= 7.5:
                            if features['Q4_price'] <= 10.5:
                                return 'Shawarma'
                            else:  # Q4_price > 10.5
                                if features['Q2_ingredients'] <= 5.5:
                                    return 'Shawarma'
                                else:  # Q2_ingredients > 5.5
                                    return 'Sushi'
                        else:  # Q2_ingredients > 7.5
                            if features['Q4_price'] <= 8.5:
                                return 'Pizza'
                            else:  # Q4_price > 8.5
                                return 'Shawarma'
                    else:  # Q4_price > 13.994999885559082
                        if features['Q4_price'] <= 16.25:
                            return 'Pizza'
                        else:  # Q4_price > 16.25
                            if features['Q3_setting_weekend_dinner'] <= 0.5:
                                return 'Shawarma'
                            else:  # Q3_setting_weekend_dinner > 0.5
                                return 'Pizza'

# Example usage
if __name__ == '__main__':
    example = {
        'id': '716549',
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': '3',
        'Q2: How many ingredients would you expect this food item to contain?': '6',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Week day lunch,At a party,Late night snack',
        'Q4: How much would you expect to pay for one serving of this food item?': '5',
        'Q5: What movie do you think of when thinking of this food item?': 'Cloudy with a Chance of Meatballs',
        'Q6: What drink would you pair with this food item?': 'Coke\xa0',
        'Q7: When you think about this food item, who does it remind you of?': 'Friends',
        'Q8: How much hot sauce would you add to this food item?': 'A little (mild)',
        'Label': 'Pizza'
    }
    prediction = predict_food(example)
    print(f'Predicted food: {prediction}')
    # You can test with other examples or without the 'Label' key to see predictions