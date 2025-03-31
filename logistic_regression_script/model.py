import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import time
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from very_good_dataloader import FoodSurveyDataLoader

def grid_search_logistic_regression(X_train, y_train, label_encoder,
                                    c_values=[0.001, 0.01, 0.1, 1, 10, 100],
                                    max_iter_values=[10, 20, 30, 40, 50],
                                    n_folds=5,
                                    output_dir=None,
                                    random_state=42):
    """
    Perform grid search to find the optimal parameters for logistic regression
    using only training data with cross-validation.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : array-like
        Training target vector
    label_encoder : object
        Encoder used for the target labels, with classes_ attribute
    c_values : list
        List of C values to search (regularization strength)
    max_iter_values : list
        List of max_iter values to search (maximum iterations)
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
    best_params = {'C': None, 'max_iter': None}
    
    # Prepare preprocessing for numerical features
    # We'll assume we need to handle both numerical and binary features
    # Get feature types based on data
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Identify binary columns (0/1 values only)
    binary_cols = []
    for col in X_train.columns:
        unique_values = X_train[col].unique()
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False, np.nan}):
            binary_cols.append(col)
    
    # Remove binary columns from numerical columns if they appear in both
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]
    
    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('binary', 'passthrough', binary_cols)
        ])
    
    # Store results for plotting
    results = []
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        f = open(f"{output_dir}/grid_search_results.txt", 'w')
        # Write header
        f.write("Grid Search Results for Logistic Regression Parameters\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of training samples: {X_train.shape[0]}\n")
        f.write(f"Number of features: {X_train.shape[1]}\n")
        f.write(f"Number of classes: {len(np.unique(y_train))}\n")
        f.write(f"Cross-validation folds: {n_folds}\n\n")
        f.write(f"{'C':<10} {'Max Iter':<10} {'Mean Accuracy':<15} {'Std Dev':<10} {'Time (s)':<10}\n")
        f.write("-" * 60 + "\n")
    
    # Fit the preprocessor once on the entire training data
    # so we don't have to refit for each parameter combination
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    # Iterate through parameter combinations
    total_combinations = len(c_values) * len(max_iter_values)
    completed = 0
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for c in c_values:
        for max_iter in max_iter_values:
            start_time = time.time()
            
            # Initialize the model with current parameters
            model = LogisticRegression(
                C=c,
                max_iter=max_iter,
                random_state=random_state
            )
            
            # Perform cross-validation on preprocessed data
            scores = cross_val_score(model, X_train_preprocessed, y_train, cv=cv, scoring='accuracy')
            
            elapsed_time = time.time() - start_time
            
            # Calculate mean and std of scores
            mean_accuracy = scores.mean()
            std_accuracy = scores.std()
            
            # Store results
            results.append({
                'C': c,
                'max_iter': max_iter,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'time': elapsed_time
            })
            
            # Write to file if output_dir is provided
            if output_dir is not None:
                f.write(f"{c:<10} {max_iter:<10} {mean_accuracy:.6f}      {std_accuracy:.6f}   {elapsed_time:.2f}\n")
            
            # Update progress
            completed += 1
            if completed % 5 == 0 or completed == total_combinations:
                print(f"Progress: {completed}/{total_combinations} parameter combinations evaluated")
            
            # Check if this is the best model so far
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = {'C': c, 'max_iter': max_iter}
    
    # After all combinations are evaluated, create the best model
    print(f"\nBest parameters: C={best_params['C']}, max_iter={best_params['max_iter']}")
    print(f"Best cross-validation accuracy: {best_accuracy:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    if output_dir is not None:
        results_df.to_csv(f"{output_dir}/grid_search_results.csv", index=False)
    
    # Create final pipeline with best parameters
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            random_state=random_state
        ))
    ])
    
    # Fit the pipeline on all training data
    best_pipeline.fit(X_train, y_train)
    best_model = best_pipeline
    
    # Generate visualizations for the best model if output_dir is provided
    if output_dir is not None:
        # Create hyperparameter visualization plots
        create_hyperparameter_plots(results_df, output_dir)
        
        # Re-run cross-validation for the best model to get predictions
        best_classifier = LogisticRegression(
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            random_state=random_state
        )
        
        # Get fold-specific scores for best model
        best_scores = cross_val_score(best_classifier, X_train_preprocessed, y_train, cv=cv, scoring='accuracy')
        
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
        plt.title(f'C={best_params["C"]}, max_iter={best_params["max_iter"]}')
        plt.ylim(min(best_scores) - 0.05, max(best_scores) + 0.05)
        plt.xticks(range(1, n_folds + 1))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{output_dir}/best_model_cv_results.png")
        plt.close()
        
        # Get predictions from cross-validation for the best model
        y_pred = cross_val_predict(best_classifier, X_train_preprocessed, y_train, cv=cv)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_train, y_pred)
        
        # Get class names if available
        if hasattr(label_encoder, 'classes_'):
            target_names = label_encoder.classes_
        else:
            target_names = [f'Class {i}' for i in range(len(np.unique(y_train)))]
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'C={best_params["C"]}, max_iter={best_params["max_iter"]}')
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
        f.write(f"Best parameters: C={best_params['C']}, max_iter={best_params['max_iter']}\n")
        f.write(f"Best cross-validation accuracy: {best_accuracy:.6f}\n")
        
        # Add fold-specific scores
        f.write("\nCross-validation scores with best hyperparameters (C={}, max_iter={}):\n".format(
            best_params['C'], best_params['max_iter']))
        f.write("-" * 60 + "\n")
        f.write("\nCross-validation scores by fold:\n")
        for i, score in enumerate(best_scores):
            f.write(f"Fold {i+1}: {score:.6f}\n")
        f.write(f"Mean accuracy: {best_scores.mean():.6f}\n")
        f.write(f"Standard deviation: {best_scores.std():.6f}\n")
        
        f.close()
    
    return best_model, best_params, best_accuracy

def create_hyperparameter_plots(results_df, output_dir):
    """
    Create plots to visualize the effect of hyperparameters on validation accuracy.
    """
    # Plot 1: C vs. Validation Accuracy
    plt.figure(figsize=(10, 6))
    c_vs_val = results_df.groupby('C')['mean_accuracy'].mean().reset_index()
    plt.plot(c_vs_val['C'], c_vs_val['mean_accuracy'], marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of C on Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_vs_validation_accuracy.png")
    plt.close()

    # Plot 2: max_iter vs. Validation Accuracy
    plt.figure(figsize=(10, 6))
    iter_vs_val = results_df.groupby('max_iter')['mean_accuracy'].mean().reset_index()
    plt.plot(iter_vs_val['max_iter'], iter_vs_val['mean_accuracy'], marker='o', linestyle='-')
    plt.xlabel('Maximum Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of Maximum Iterations on Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/max_iter_vs_validation_accuracy.png")
    plt.close()

    # Plot 3: C vs. Validation Accuracy for different max_iter values
    plt.figure(figsize=(12, 8))
    for max_iter in sorted(results_df['max_iter'].unique()):
        subset = results_df[results_df['max_iter'] == max_iter]
        if not subset.empty:
            c_vs_val_subset = subset.groupby('C')['mean_accuracy'].mean().reset_index()
            plt.plot(c_vs_val_subset['C'], c_vs_val_subset['mean_accuracy'], marker='o', linestyle='-', 
                    label=f'max_iter={max_iter}')
    plt.xscale('log')
    plt.xlabel('C (Regularization Strength)')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of C on Validation Accuracy by max_iter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_vs_validation_accuracy_by_max_iter.png")
    plt.close()

    # Plot 4: Heatmap of C vs max_iter on validation accuracy
    plt.figure(figsize=(12, 8))
    pivot_table = results_df.pivot_table(values='mean_accuracy', index='max_iter', columns='C', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
    plt.ylabel('max_iter')
    plt.xlabel('C')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_max_iter_validation_heatmap.png")
    plt.close()

    # Plot 5: Training time for different parameter combinations
    plt.figure(figsize=(12, 8))
    pivot_time = results_df.pivot_table(values='time', index='max_iter', columns='C', aggfunc='mean')
    sns.heatmap(pivot_time, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Training Time (seconds) by C and max_iter')
    plt.ylabel('max_iter')
    plt.xlabel('C')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/c_max_iter_time_heatmap.png")
    plt.close()

def main():
    """
    Main function to run the grid search for logistic regression.
    """
    # Load and preprocess data
    file_path = 'cleaned_data_combined_modified.csv'
    output_dir = 'logistic_regression_cv_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data using the data loader
    dataloader = FoodSurveyDataLoader(file_path)
    dataloader.load_data()
    df = dataloader.preprocess_data()
    
    # Get features and labels
    X = df[dataloader.feature_names]
    y = df['label_encoded']
    
    # Split into train and test sets to avoid data leakage
    X_train, X_test, y_train, y_test = dataloader.stratified_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Run grid search with cross-validation
    best_model, best_params, best_accuracy = grid_search_logistic_regression(
        X_train, 
        y_train,
        dataloader,  # Pass the dataloader to access index_to_class mapping
        c_values=[0.001, 0.01, 0.1, 1, 10, 100],
        max_iter_values=range(10, 101, 10),
        n_folds=5,
        output_dir=output_dir,
        random_state=42
    )
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTest Accuracy with best model: {test_accuracy:.4f}")
    
    # Create test set confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Get class names
    if hasattr(dataloader, 'index_to_class'):
        target_names = [dataloader.index_to_class[i] for i in range(len(np.unique(y)))]
    else:
        target_names = [f'Class {i}' for i in range(len(np.unique(y)))]

    # Add classification report
    with open(f"{output_dir}/grid_search_results.txt", 'a') as f:
        f.write("\nClassification Report (Cross-Validation):\n")
        report = classification_report(y_test, y_test_pred, target_names=target_names)
        f.write(report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Set Confusion Matrix')
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
    plt.savefig(f"{output_dir}/test_confusion_matrix.png")
    plt.close()
    
    # Save the model components separately as required by load_model_components()
    
    # 1. Extract and save preprocessor
    preprocessor = best_model.named_steps['preprocessor']
    with open(f"{output_dir}/preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # 2. Extract and save classifier
    classifier = best_model.named_steps['classifier']
    with open(f"{output_dir}/classifier.pkl", 'wb') as f:
        pickle.dump(classifier, f)
    
    # 3. Create and save feature lists dictionary
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Identify binary columns (0/1 values only)
    binary_cols = []
    for col in X_train.columns:
        unique_values = X_train[col].unique()
        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False, np.nan}):
            binary_cols.append(col)
    
    # Remove binary columns from numerical columns if they appear in both
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]
    
    feature_lists = {
        'features': list(X_train.columns),
        'numerical_cols': numerical_cols,
        'binary_cols': binary_cols
    }
    
    with open(f"{output_dir}/feature_lists.pkl", 'wb') as f:
        pickle.dump(feature_lists, f)
    
    # Also save the full model as before for backwards compatibility
    with open(f"{output_dir}/best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Model saved to {output_dir}/best_model.pkl")
    print(f"Model components saved to:")
    print(f"  - {output_dir}/preprocessor.pkl")
    print(f"  - {output_dir}/classifier.pkl")
    print(f"  - {output_dir}/feature_lists.pkl")
    print(f"Results and visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()