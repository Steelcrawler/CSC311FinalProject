import pandas as pd
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os

def extract_scale(text):
    if pd.isna(text):
        return None
    match = re.search(r'(\d+)', str(text))
    if match:
        num = int(match.group(1))
        if 1 <= num <= 5:
            return num
    return None

def extract_ingredients(text):
    if pd.isna(text):
        return None
    match = re.search(r'(\d+)', str(text))
    if match:
        return int(match.group(1))
    return str(text).count(',') + 1

def text_to_number(text):
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

def extract_price(text):
    if pd.isna(text):
        return None
    
    text = text_to_number(str(text).lower())

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

def extract_common_words(text):
    if pd.isna(text) or not isinstance(text, str):
        return Counter()
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    stop_words = {
        'the', 'a', 'an', 'in', 'of', 'to', 'and', 'is', 'it', 'this', 'that', 'would', 'with', 
        'for', 'on', 'at', 'be', 'i', 'food', 'item', 'my', 'me', 'you', 'your', 'they', 'their'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    return Counter(filtered_words)

def parse_hot_sauce_amount(text, count_nan_as_none=True):
    if pd.isna(text) or not isinstance(text, str):
        return 'none' if count_nan_as_none else None
    
    # Define stop words to ignore when matching
    stop_words = {
        'the', 'a', 'an', 'in', 'of', 'to', 'and', 'is', 'it', 'this', 'that', 'would', 'with', 
        'for', 'on', 'at', 'be', 'i', 'food', 'item', 'my', 'me', 'you', 'your', 'they', 'their'
    }

    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    
    categories = {
        'none': ['none', 'no', 'zero', 'not', "don't", 'never', 'wouldn\'t', '0', 'nothing'],
        'little': ['little', 'bit', 'dash', 'drop', 'hint', 'splash', 'tiny', 'small', 'minimal', 'light', 'mild'],
        'some': ['some', 'medium', 'moderate', 'regular', 'normal', 'average'],
        'lot': ['lot', 'lots', 'much', 'plenty', 'generous', 'good amount', 'spicy', 'extra', 'very', 'maximum', 
                'tons', 'drowning', 'all', 'heavy', 'doused', 'covered', 'drenched', 'hot']
    }
    
    for category, keywords in categories.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', filtered_text) for keyword in keywords):
            return category

    for category, keywords in categories.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords):
            return category
            
    return 'other'

def analyze_food_survey(csv_path):
    df = pd.read_csv(csv_path)
    
    df['Q1_scale'] = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].apply(extract_scale)
    df['Q2_ingredients'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(extract_ingredients)
    df['Q4_price'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(extract_price)
    df['Q8_hot_sauce'] = df['Q8: How much hot sauce would you add to this food item?'].apply(parse_hot_sauce_amount)
    
    # Mark unparseable entries
    df['Q1_unparseable'] = df['Q1_scale'].isna()
    df['Q2_unparseable'] = df['Q2_ingredients'].isna()
    df['Q4_unparseable'] = df['Q4_price'].isna()
    df['Q8_unparseable'] = df['Q8_hot_sauce'].isna()
    
    text_columns = [
        'Q3: In what setting would you expect this food to be served? Please check all that apply',
        'Q5: What movie do you think of when thinking of this food item?',
        'Q6: What drink would you pair with this food item?',
        'Q7: When you think about this food item, who does it remind you of?'
    ]
    
    for col in text_columns:
        short_col = col.split(':')[0]  
        df[f'{short_col}_unparseable'] = df[col].apply(
            lambda x: pd.isna(x) or not isinstance(x, str) or len(extract_common_words(x)) == 0
        )

    labels = df['Label'].unique()
    results = {}
    results['unparseable'] = {
        'Q1': {label: int(df[df['Label'] == label]['Q1_unparseable'].sum()) for label in labels},
        'Q2': {label: int(df[df['Label'] == label]['Q2_unparseable'].sum()) for label in labels},
        'Q3': {label: int(df[df['Label'] == label]['Q3_unparseable'].sum()) for label in labels},
        'Q4': {label: int(df[df['Label'] == label]['Q4_unparseable'].sum()) for label in labels},
        'Q5': {label: int(df[df['Label'] == label]['Q5_unparseable'].sum()) for label in labels},
        'Q6': {label: int(df[df['Label'] == label]['Q6_unparseable'].sum()) for label in labels},
        'Q7': {label: int(df[df['Label'] == label]['Q7_unparseable'].sum()) for label in labels},
        'Q8': {label: int(df[df['Label'] == label]['Q8_unparseable'].sum()) for label in labels}
    }
    
    # Total unparseable counts by question
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']:
        results['unparseable'][q]['total'] = sum(results['unparseable'][q].values())
    
    # Q1: Complexity scale
    results['Q1_complexity'] = {label: list(df[df['Label'] == label]['Q1_scale'].dropna()) for label in labels}

    # Q2: Ingredient count - store all values
    results['Q2_ingredients'] = {label: list(df[df['Label'] == label]['Q2_ingredients'].dropna()) for label in labels}
    
    # Q3: Setting
    results['Q3_setting'] = {label: Counter() for label in labels}
    for label in labels:
        for text in df[df['Label'] == label]['Q3: In what setting would you expect this food to be served? Please check all that apply']:
            word_counts = extract_common_words(text)
            for word, count in word_counts.items():
                results['Q3_setting'][label][word] += count
    
    # Q4: Price
    results['Q4_price'] = {label: list(df[df['Label'] == label]['Q4_price'].dropna()) for label in labels}
    
    # Q5: Movie
    results['Q5_movie'] = {label: Counter() for label in labels}
    for label in labels:
        for text in df[df['Label'] == label]['Q5: What movie do you think of when thinking of this food item?']:
            word_counts = extract_common_words(text)
            for word, count in word_counts.items():
                results['Q5_movie'][label][word] += count
    
    # Q6: Drink
    results['Q6_drink'] = {label: Counter() for label in labels}
    for label in labels:
        for text in df[df['Label'] == label]['Q6: What drink would you pair with this food item?']:
            word_counts = extract_common_words(text)
            for word, count in word_counts.items():
                results['Q6_drink'][label][word] += count
    
    # Q7: Reminds of who
    results['Q7_person'] = {label: Counter() for label in labels}
    for label in labels:
        for text in df[df['Label'] == label]['Q7: When you think about this food item, who does it remind you of?']:
            word_counts = extract_common_words(text)
            for word, count in word_counts.items():
                results['Q7_person'][label][word] += count
    
    # Q8: Hot sauce 
    results['Q8_hot_sauce'] = {label: df[df['Label'] == label]['Q8_hot_sauce'].value_counts().to_dict() for label in labels}
    
    return results, df

def print_results(results):
    print("\n=== Unparseable Responses ===")
    for question, data in results['unparseable'].items():
        print(f"\n{question}:")
        for label, count in data.items():
            print(f"  {label}: {count}")
    
    for question, data in results.items():
        if question == 'unparseable':
            continue
            
        print(f"\n=== {question} ===")
        for label, stats in data.items():
            print(f"\n{label}:")
            if question == 'Q4_price':
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        print(f"  {stat_name}: {value:.2f}")
                    else:
                        print(f"  {stat_name}: {value}")
            else:
                if isinstance(stats, dict):
                    for item, count in list(stats.items())[:5]:  # Top 5 items
                        print(f"  {item}: {count}")
                else:
                    print(f"  {stats}")


def plot_word_histograms(results, question_number, folder_dir, num_words=20):
    """
    Create histograms of top words/values for a specific question, broken down by label.
    
    Args:
        results (dict): Results dictionary from analyze_food_survey
        question_number (int): Question number (e.g., 3 for Q3)
        folder_dir (str): Directory to save plots
        num_words (int): Number of top words/values to display
    """
    
    os.makedirs(folder_dir, exist_ok=True)
    q_key = f"Q{question_number}_"
    
    # Find the matching key in results
    matching_keys = [k for k in results.keys() if k.startswith(q_key)]
    
    if not matching_keys:
        print(f"Question {question_number} not found in results.")
        return
        
    q_key = matching_keys[0]
    q_data = results[q_key]
    
    print(f"\nCreating histograms for {q_key}:")
    
    if q_key in ["Q1_complexity", "Q2_ingredients", "Q4_price"]:
        # First determine global min, max for consistent scale
        all_values = []
        for label, values in q_data.items():
            all_values.extend(values)
        
        if not all_values:
            print(f"No data available for question {question_number}")
            return
            
        all_values = np.array(all_values)
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        
        # Determine bins based on question type
        if q_key == "Q1_complexity":
            # For Q1 (scale 1-5), use 5 bins
            bins = np.arange(0.5, 6.5, 1)
            x_min, x_max = 0.5, 5.5
            plt_xlabel = "Complexity Scale (1-5)"
            title_desc = "Complexity Rating"
        elif q_key == "Q2_ingredients":
            # For Q2, integer bins for ingredient counts
            max_ingredients = int(np.ceil(global_max))
            bins = np.arange(0.5, max_ingredients + 1.5, 1)
            x_min, x_max = 0.5, max_ingredients + 0.5
            plt_xlabel = "Number of Ingredients"
            title_desc = "Ingredient Count"
        else:  # Q4_price
            # For Q4, create reasonable bins
            bin_width = (global_max - global_min) / min(20, max(5, int(np.sqrt(len(all_values)))))
            bins = np.arange(global_min, global_max + bin_width, bin_width)
            x_min, x_max = global_min - bin_width, global_max + bin_width
            plt_xlabel = "Price ($)"
            title_desc = "Price Distribution"
        
        # Now plot each label with consistent binning and scale
        for label, values in q_data.items():
            plt.figure(figsize=(10, 6))
            
            if len(values) > 0:
                # Convert to numpy array if not already
                values = np.array(values)
                
                # Plot the histogram with consistent bins
                plt.hist(values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Add vertical lines for mean and median
                mean_val = np.mean(values)
                plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, 
                            label=f'Mean: {mean_val:.2f}')
                
                median_val = np.median(values)
                plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1,
                           label=f'Median: {median_val:.2f}')
                
                # Set consistent x-axis limits
                plt.xlim(x_min, x_max)
                
                # Add labels and title
                plt.title(f"{title_desc} for {label} - Question {question_number}")
                plt.xlabel(plt_xlabel)
                plt.ylabel("Frequency")
                plt.legend()
                
                # For Q1 and Q2, set integer ticks
                if q_key in ["Q1_complexity", "Q2_ingredients"]:
                    plt.xticks(range(1, int(global_max)+1))
            else:
                plt.text(0.5, 0.5, "No data available", ha='center', va='center')
                
            plt.tight_layout()
            filename = f"q{question_number}_{label.lower()}_{num_words}.png"
            plt.savefig(os.path.join(folder_dir, filename))
            plt.close()
        return
    
    for label, data in q_data.items():
        plt.figure(figsize=(14, 8))
        
        # Extract top words and counts based on data type
        words = []
        counts = []
        
        if isinstance(data, Counter):  
            # print(f"  {label}: {len(data)} unique words/values available")
            
            # Get the most common N words
            top_words = data.most_common(num_words)
            words = [str(w) for w, _ in top_words]
            counts = [c for _, c in top_words]
            
            # print(f"    Plotting top {len(words)} words/values")
            
        elif isinstance(data, dict):
            # print(f"  {label}: {len(data)} unique categories/values available")

            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            top_items = sorted_items[:min(num_words, len(sorted_items))]
            words = [str(w) for w, _ in top_items]
            counts = [c for _, c in top_items]

            # print(f"    Plotting top {len(words)} categories/values")
            
        else:
            # print(f"Unsupported data type for {label} in question {question_number}")
            plt.close()
            continue
        
        if not words:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
        else:
            bars = plt.bar(words, counts)
            plt.title(f"Top {len(words)} Values for {label} - Question {question_number}")
            plt.xlabel("Words/Values")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha="right")
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}',
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        filename = f"q{question_number}_{label.lower()}_{len(words)}.png"
        full_path = os.path.join(folder_dir, filename)
        plt.savefig(full_path)
        print(f"    Saved to {full_path}")
        plt.close()
        
    print(f"Plots for Question {question_number} saved to {folder_dir}")
        

def main():
    csv_path = 'cleaned_data_combined_modified.csv'
    
    results, df = analyze_food_survey(csv_path)
    # print_results(results)
    
    for i in range(1, 9):
        plot_word_histograms(results, i, "word_histograms", num_words=20)
    # return results, df

if __name__ == "__main__":
    main()