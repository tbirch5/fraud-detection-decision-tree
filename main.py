from src.preprocess import load_and_preprocess_data
from src.model import train_model, evaluate_model, visualize_tree
import pandas as pd

print("🔄 Starting data preprocessing...")
X_train, X_test, y_train, y_test = load_and_preprocess_data()
print("✅ Data preprocessing complete.")

print("🌲 Training decision tree model...")
model = train_model(X_train, y_train)
print("✅ Model training complete.")

print("🧪 Evaluating model...")
evaluate_model(model, X_test, y_test)
print("✅ Evaluation complete.")

# Load feature names for the tree plot
df = pd.read_csv('./data/creditcard.csv')
feature_names = df.drop('Class', axis=1).columns.tolist()

print("🌳 Generating Decision Tree visualization...")
visualize_tree(model, feature_names)
print("✅ Tree visualization completed.")