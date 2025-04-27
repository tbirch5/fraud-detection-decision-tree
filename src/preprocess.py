import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def load_and_preprocess_data(path='./data/creditcard.csv'):
    # Resolve absolute path in case it's run from another directory
    path = os.path.abspath(path)
    print(f"📥 Loading dataset from: {path}")

    # Load data
    df = pd.read_csv(path)
    print(f"✅ Data loaded. Shape: {df.shape}")

    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance the training data with SMOTE
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

    print(f"📊 Resampled training set shape: {X_train_resampled.shape}, {y_train_resampled.shape}")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test
