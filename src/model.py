from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def visualize_tree(clf, feature_names):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    import os
    # Make sure reports directory exists
    current_dir = os.path.dirname(__file__)
    reports_dir = os.path.abspath(os.path.join(current_dir, '..', 'reports'))
    os.makedirs(reports_dir, exist_ok=True)

    #Generate and save tree plot
    plt.figure(figsize=(20,10))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=["Non-Fraud", "Fraud"], max_depth=3, fontsize=10)
    
    tree_path = os.path.join(reports_dir, 'tree_visualization.png')
    plt.savefig(tree_path)
    print(f"🌳 Decision Tree plot saved to: {tree_path}")
    plt.show()

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    # Print Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
   

    # Create reports directory if it doesn't exist
    current_dir = os.path.dirname(__file__)
    reports_dir = os.path.abspath(os.path.join(current_dir, '..', 'reports'))
    os.makedirs(reports_dir, exist_ok=True)

    save_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(save_path)

    print(f"✅ Confusion matrix saved to: {save_path}")
    # plt.show()
"""
plt.show() has been commented out for the confution matrix due to user interaction required when 
running main.py. if matrix diagram not closed other diagrams will not be generated until the diagram
box is closed.
"""
