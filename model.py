"""
ML Model Training and Evaluation Script
Dataset: Iris Flower Dataset
Model: Random Forest Classifier
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """Load the Iris dataset and split into train and test sets."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, iris.target_names


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Make predictions and compute accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc


def main():
    print("==================================================")
    print("   ML CI/CD Pipeline - Iris Classification Model   ")
    print("==================================================")

    # 1. Load Dataset
    print("\n[1] Loading Iris Dataset...")
    X_train, X_test, y_train, y_test, target_names = load_data()
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Testing samples : {X_test.shape[0]}")
    print(f"    Target classes  : {list(target_names)}")

    # 2. Train Model
    print("\n[2] Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    print("    Training completed successfully.")

    # 3. Make Predictions & Evaluate
    print("\n[3] Making Predictions and Evaluating Model...")
    y_pred, accuracy = evaluate_model(model, X_test, y_test)

    print("\n==================================================")
    print(f"Model Name    : Random Forest Classifier")
    print(f"Dataset Used  : Iris Flower Dataset")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("==================================================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return accuracy


if __name__ == "__main__":
    main()
