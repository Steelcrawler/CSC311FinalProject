import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from very_good_dataloader import FoodSurveyDataLoader
from model_training import grid_search_decision_tree, evaluate_model

def main():
    """Main function to train and evaluate the model"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a decision tree model on food survey data')
    parser.add_argument('--data', type=str, default='../cleaned_data_combined_modified.csv', help='Path to the CSV data file')
    parser.add_argument('--output', type=str, default='food_survey_results', help='Output directory for results')
    parser.add_argument('--min-depth', type=int, default=1, help='Minimum tree depth to try')
    parser.add_argument('--max-depth', type=int, default=15, help='Maximum tree depth to try')
    parser.add_argument('--min-samples', type=int, default=2, help='Minimum samples split to try')
    parser.add_argument('--max-samples', type=int, default=20, help='Maximum samples split to try')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    output_dir = f"{args.output}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting model training with data from {args.data}")
    print(f"Results will be saved to {output_dir}")
    
    # Load and preprocess data
    print("\n=== Loading and preprocessing data ===")
    data_loader = FoodSurveyDataLoader(args.data)
    df = data_loader.preprocess_data()
    
    # Get feature groups and print summary
    feature_groups = data_loader.get_feature_groups()
    
    # Write feature groups to a file
    with open(f"{output_dir}/feature_groups.txt", 'w') as f:
        f.write("Feature Groups Summary:\n")
        f.write("=" * 30 + "\n\n")
        for group, features in feature_groups.items():
            f.write(f"{group}: {len(features)} features\n")
            if len(features) <= 10:
                f.write(f"  {features}\n")
            else:
                f.write(f"  First 5: {features[:5]}\n")
                f.write(f"  Last 5: {features[-5:]}\n")
            f.write("\n")
    
    # Split data
    print("\n=== Splitting data into train and test sets ===")
    X_train, X_test, y_train, y_test, feature_names, label_encoder = data_loader.split_data(
        test_size=0.2, random_state=args.seed
    )
    
    # Perform grid search
    print("\n=== Performing grid search for optimal parameters ===")
    best_model, best_params, best_cv_accuracy = grid_search_decision_tree(
        X_train, y_train, label_encoder,
        max_depth_range=(args.min_depth, args.max_depth),
        min_samples_split_range=(args.min_samples, args.max_samples),
        n_folds=args.folds,
        output_dir=output_dir,
        random_state=args.seed
    )
    
    # Evaluate on test data
    print("\n=== Evaluating model on test data ===")
    accuracy, report, cm = evaluate_model(
        best_model, X_test, y_test, label_encoder, output_dir=output_dir
    )
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Best parameters: max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
    print(f"Cross-validation accuracy: {best_cv_accuracy:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"All results and visualizations saved to {output_dir}")
    
    # Save summary to file
    with open(f"{output_dir}/training_summary.txt", 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Data file: {args.data}\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Testing samples: {X_test.shape[0]}\n")
        f.write(f"Number of features: {X_train.shape[1]}\n")
        f.write(f"Best parameters: max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}\n")
        f.write(f"Cross-validation accuracy: {best_cv_accuracy:.4f}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")

if __name__ == "__main__":
    main()