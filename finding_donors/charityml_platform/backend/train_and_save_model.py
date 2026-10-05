"""
CharityML Model Training & Export Script
Trains the optimized Gradient Boosting Classifier on census.csv
and saves the model, scaler, and column schema for the Flask API.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_and_export():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Path to census.csv (in the parent directory)
    csv_path = os.path.join(current_dir, "..", "..", "census.csv")
    if not os.path.exists(csv_path):
        # Fallback to local check
        csv_path = os.path.join(current_dir, "..", "census.csv")
    
    print(f"Loading data from: {csv_path}")
    data = pd.read_csv(csv_path)
    
    # Split raw target and features
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)
    
    # Log-transform skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log = pd.DataFrame(data=features_raw)
    features_log[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    
    # MinMax scaling on numerical features
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = MinMaxScaler()
    features_log[numerical] = scaler.fit_transform(features_log[numerical])
    
    # One-hot encoding
    features_final = pd.get_dummies(features_log)
    column_names = list(features_final.columns)
    
    # Encode target
    y = income_raw.apply(lambda x: 1 if x == '>50K' else 0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features_final, y, test_size=0.2, random_state=0
    )
    
    print("Training optimized GradientBoostingClassifier (n_estimators=150, lr=0.1)...")
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Save artifacts
    bundle = {
        'model': clf,
        'scaler': scaler,
        'numerical_cols': numerical,
        'feature_columns': column_names,
        'skewed_cols': skewed
    }
    
    output_file = os.path.join(model_dir, "charityml_model.joblib")
    joblib.dump(bundle, output_file)
    print(f"Model and artifacts successfully saved to: {output_file}")
    print(f"Total features: {len(column_names)}")

if __name__ == "__main__":
    train_and_export()
