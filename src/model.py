from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def train_model(X_train, y_train):
    """
    Train a Decision Tree Classifier using GridSearchCV to optimize for F1 score.
    """
    # Define hyperparameter grid
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'class_weight': [None, 'balanced']
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42), 
        params, 
        scoring='f1', 
        cv=5, 
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"✅ Best Parameters: {grid.best_params_}")
    return best_model

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the model using an automaticall selected optimal threshold for best F1 score.
    Also generates and saves confusion matrix and precision-recall curve plots.
    """

    y_proba = clf.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6) # Avoid division by zero
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    # Predictions using best threshold
    y_pred = (y_proba > best_threshold).astype(int)


    # Print Metrics
    print(f"📊 Best Threshold: {best_threshold:.2f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Setup reports directory
    current_dir = os.path.dirname(__file__)
    reports_dir = os.path.abspath(os.path.join(current_dir, '..', 'reports'))
    os.makedirs(reports_dir, exist_ok=True)

    # Plot confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'], 
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"✅ Confusion matrix saved to: {cm_path}")


    # Precision-Recall vs Threshold plot
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(reports_dir, 'precision_recall_threshold.png')
    plt.savefig(pr_path)
    print(f"Precision-Recall plot saved to: {pr_path}")

    return best_threshold, f1_score(y_test, y_pred)


def visualize_tree(clf, feature_names):
    """Visualizes and saves a plot of the decision tree.
    """
    plt.figure(figsize=(20,10))
    plot_tree(
        clf, 
        filled=True, 
        feature_names=feature_names, 
        class_names=["Non-Fraud", "Fraud"], 
        max_depth=3, 
        fontsize=10
    )
    
    current_dir = os.path.dirname(__file__)
    reports_dir = os.path.abspath(os.path.join(current_dir, '..', 'reports'))
    os.makedirs(reports_dir, exist_ok=True)
    
    tree_path = os.path.join(reports_dir, 'tree_visualization.png')
    plt.savefig(tree_path)
    print(f"🌳 Decision Tree plot saved to: {tree_path}")
    # plt.show()
