import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os

def grid_search_decision_tree(X_train, y_train, label_encoder, 
                             max_depth_range=(1, 20), 
                             min_samples_split_range=(2, 40), 
                             n_folds=5, 
                             output_dir=None, 
                             random_state=42):
    """
    Perform grid search to find the optimal parameters for the decision tree
    using only training data with cross-validation.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : array-like
        Training target vector
    label_encoder : object
        Encoder used for the target labels, with classes_ attribute
    max_depth_range : tuple
        (min_depth, max_depth) range to search
    min_samples_split_range : tuple
        (min_split, max_split) range to search
    n_folds : int
        Number of folds for cross-validation
    output_dir : str or None
        Directory to save output files. If None, no files are saved.
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (best_model, best_params, best_accuracy)
    """
    # Initialize variables to track best performance
    best_accuracy = 0
    best_model = None
    best_params = {'max_depth': None, 'min_samples_split': None}
    
    # Prepare ranges
    min_depth, max_depth = max_depth_range
    min_split, max_split = min_samples_split_range
    
    # Store results for plotting
    results = []
    
    # Open file for saving results if output_dir is provided
    if output_dir is not None:
        f = open(f"{output_dir}/grid_search_results.txt", 'w')
        # Write header
        f.write("Grid Search Results for Decision Tree Parameters\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of training samples: {X_train.shape[0]}\n")
        f.write(f"Number of features: {X_train.shape[1]}\n")
        f.write(f"Number of classes: {len(np.unique(y_train))}\n")
        f.write(f"Cross-validation folds: {n_folds}\n\n")
        f.write(f"{'Max Depth':<10} {'Min Samples Split':<20} {'Mean Accuracy':<15} {'Std Dev':<10} {'Time (s)':<10}\n")
        f.write("-" * 60 + "\n")
    
    # Iterate through parameter combinations
    total_combinations = (max_depth - min_depth + 1) * (max_split - min_split + 1)
    completed = 0
    
    for depth in range(min_depth, max_depth + 1):
        for min_samples in range(min_split, max_split + 1):
            start_time = time.time()
            
            # Initialize the model with current parameters
            model = DecisionTreeClassifier(
                max_depth=depth if depth > 0 else None, 
                min_samples_split=min_samples,
                random_state=random_state
            )
            
            # Initialize cross-validation
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            elapsed_time = time.time() - start_time
            
            # Calculate mean and std of scores
            mean_accuracy = scores.mean()
            std_accuracy = scores.std()
            
            # Store results
            results.append({
                'max_depth': depth,
                'min_samples_split': min_samples,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'time': elapsed_time
            })
            
            # Write to file if output_dir is provided
            if output_dir is not None:
                f.write(f"{depth:<10} {min_samples:<20} {mean_accuracy:.6f}      {std_accuracy:.6f}   {elapsed_time:.2f}\n")
            
            # Update progress
            completed += 1
            if completed % 10 == 0 or completed == total_combinations:
                print(f"Progress: {completed}/{total_combinations} parameter combinations evaluated")
            
            # Check if this is the best model so far
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = {'max_depth': depth, 'min_samples_split': min_samples}
    
    # After all combinations are evaluated, train the model with the best parameters
    print(f"\nBest parameters: max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
    print(f"Best cross-validation accuracy: {best_accuracy:.4f}")
    
    # Train the final model with the best parameters using ALL training data
    best_model = DecisionTreeClassifier(
        max_depth=best_params['max_depth'] if best_params['max_depth'] > 0 else None,
        min_samples_split=best_params['min_samples_split'],
        random_state=random_state
    )
    best_model.fit(X_train, y_train)
    
    # Generate the confusion matrix only for the best model using cross-validation on training data
    if output_dir is not None:
        # Re-run cross-validation for the best model to get fold-specific scores
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        best_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Create a bar chart of cross-validation results for the best model
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(1, n_folds + 1), best_scores, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axhline(best_scores.mean(), color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean Accuracy: {best_scores.mean():.4f}')
        
        # Add text labels above each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{best_scores[i]:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title(f'Cross-Validation Results for Best Model\nmax_depth={best_params["max_depth"]}, min_samples_split={best_params["min_samples_split"]}')
        plt.ylim(min(best_scores) - 0.05, max(best_scores) + 0.05)
        plt.xticks(range(1, n_folds + 1))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/best_model_cv_results.png")
        plt.close()
        
        # Get predictions from cross-validation for the best model
        y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_train, y_pred)
        target_names = label_encoder.classes_
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Cross-Validation Confusion Matrix (Best Model)\nmax_depth={best_params["max_depth"]}, min_samples_split={best_params["min_samples_split"]}')
        plt.colorbar()
        
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
        
        # Add text annotations
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
        plt.savefig(f"{output_dir}/cv_confusion_matrix.png")
        plt.close()
        
        # Write summary to the results file
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"Best parameters: max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}\n")
        f.write(f"Best cross-validation accuracy: {best_accuracy:.6f}\n")
        
        # Add fold-specific scores
        f.write("\nCross-validation scores by fold:\n")
        for i, score in enumerate(best_scores):
            f.write(f"Fold {i+1}: {score:.6f}\n")
        f.write(f"Standard deviation: {best_scores.std():.6f}\n")
        
        f.close()
        
        # Create a heatmap of the grid search results
        results_df = pd.DataFrame(results)
        pivot_table = results_df.pivot_table(
            index='min_samples_split', 
            columns='max_depth', 
            values='mean_accuracy'
        )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pivot_table, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Mean Accuracy')
        plt.title('Grid Search Results')
        plt.xlabel('Max Depth')
        plt.ylabel('Min Samples Split')
        
        # Adjust x and y ticks
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        
        # Mark the best parameters
        best_idx = (pivot_table.index.get_loc(best_params['min_samples_split']), 
                   pivot_table.columns.get_loc(best_params['max_depth']))
        plt.plot(best_idx[1], best_idx[0], 'r*', markersize=15)
        
        # Save the heatmap
        plt.savefig(f"{output_dir}/grid_search_heatmap.png")
        plt.close()
    
    return best_model, best_params, best_accuracy

def evaluate_model(model, X_test, y_test, label_encoder, output_dir=None):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained decision tree model
    X_test : pandas.DataFrame
        Test feature matrix
    y_test : array-like
        Test target vector
    label_encoder : object
        Encoder used for the target labels, with classes_ attribute
    output_dir : str or None
        Directory to save output files. If None, no files are saved.
        
    Returns:
    --------
    tuple
        (accuracy, classification_report_text, confusion_matrix)
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    target_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize confusion matrix on test data if output_dir is provided
    if output_dir is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Test Data Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/test_confusion_matrix.png")
        plt.close()
        
        # Also save a text report
        with open(f"{output_dir}/test_evaluation.txt", 'w') as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            
        # Visualize feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance (Test Model)')
            n_features = min(20, len(feature_names))
            plt.bar(range(n_features), importances[indices[:n_features]])
            plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png")
            plt.close()
            
            # Save feature importance to text file
            with open(f"{output_dir}/feature_importance.txt", 'w') as f:
                f.write("Feature Importance:\n")
                f.write("-" * 30 + "\n")
                for i in range(min(20, len(feature_names))):
                    idx = indices[i]
                    f.write(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}\n")
    
    return accuracy, report, cm

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for training and evaluating decision tree models.")
    print("Import the functions in your own script to use them.")