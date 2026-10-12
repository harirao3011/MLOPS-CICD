"""
Automated Test Cases for Machine Learning Model Pipeline
Framework: pytest
"""

import os
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import load_data, train_model, evaluate_model


def test_data_loading():
    """Verify that dataset loads properly with expected shapes and classes."""
    X_train, X_test, y_train, y_test, target_names = load_data()

    assert X_train.shape[0] > 0, "Training set should not be empty"
    assert X_test.shape[0] > 0, "Test set should not be empty"
    assert X_train.shape[1] == 4, "Iris dataset must have 4 features"
    assert X_test.shape[1] == 4, "Iris dataset must have 4 features"
    assert len(y_train) == X_train.shape[0], "Feature and label counts must match"
    assert len(y_test) == X_test.shape[0], "Feature and label counts must match"
    assert len(target_names) == 3, "Iris dataset must contain 3 target classes"


def test_model_training():
    """Verify model instantiation and fitting."""
    X_train, _, y_train, _, _ = load_data()
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"
    assert hasattr(model, "estimators_"), "Model should be fitted and contain estimators"
    assert len(model.estimators_) == 50, "Model should have 50 estimators"


def test_model_predictions():
    """Verify that model predictions are valid and correspond to test set dimensions."""
    X_train, X_test, y_train, y_test, _ = load_data()
    model = train_model(X_train, y_train)
    y_pred, _ = evaluate_model(model, X_test, y_test)

    assert len(y_pred) == len(y_test), "Predictions count must match test sample count"
    assert set(np.unique(y_pred)).issubset({0, 1, 2}), "Predictions must be valid class labels (0, 1, 2)"


def test_model_accuracy_threshold():
    """Verify that model accuracy meets or exceeds the required 90% threshold."""
    X_train, X_test, y_train, y_test, _ = load_data()
    model = train_model(X_train, y_train)
    _, accuracy = evaluate_model(model, X_test, y_test)

    print(f"\nEvaluated Model Accuracy: {accuracy * 100:.2f}%")
    assert accuracy >= 0.90, f"Model accuracy ({accuracy * 100:.2f}%) must be >= 90%"


def test_index_html_metadata():
    """Verify that index.html exists and contains the required assignment fields."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    assert os.path.exists(html_path), "index.html file must exist"

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Random Forest" in content, "index.html must display the ML model name"
    assert "Iris" in content, "index.html must display the dataset used"
    assert "Accuracy" in content, "index.html must display the model accuracy"
    assert "CI/CD" in content or "Deployment Status" in content, "index.html must display CI/CD deployment status"
