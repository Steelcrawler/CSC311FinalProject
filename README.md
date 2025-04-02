# CSC311 Food Classification Project!

Hello! This is our CSC311 Classification Project. Below is how to interpret the repo.

## Repository Structure

- **very_good_dataloader.py**: The core data loading and preprocessing utility used across all model training. This dataloader handles complex parsing of various question types from our food survey, including numeric ratings, ingredient counts, categorical settings, price values, and text responses using bag-of-words approaches.

- **decision_tree/**: Contains all files related to our decision tree model
  - **model_training.py**: Core functionality for training decision tree models
  - **train.py**: Script to execute decision tree training
  - **food_survey_results/**: Directory containing saved model outputs and performance metrics

- **logistic_regression/**: Contains all code for training and evaluating our logistic regression models
  - Includes implementation files for multinomial logistic regression classifier

- **neural_net/**: Contains implementation and training files for our neural network models
  - Includes model architecture definition and training scripts

- **EDA.py**: Exploratory Data Analysis script that generated visualizations
  - **word_histograms/**: Directory containing generated figures from our data analysis
    - Includes frequency distributions and other visualizations used in our report

## Data Processing

Our dataloader implements sophisticated preprocessing techniques for the food survey data:
- Complexity ratings (Q1): Extracted using regex and one-hot encoded
- Ingredient counts (Q2): Converted from text to numeric values
- Settings (Q3): Parsed checkbox responses into binary features
- Price values (Q4): Standardized various price formats into floating-point values
- Text responses (Q5-Q7): Processed using custom bag-of-words with frequency filtering
- Hot sauce preference (Q8): Treated as categorical with one-hot encoding
