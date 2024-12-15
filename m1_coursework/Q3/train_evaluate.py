from sklearn.metrics import accuracy_score

def train_and_evaluate(classifier, X_train, X_val, y_train, y_val):
    """
    Train a classifier and evaluate its accuracy.
    Args:
        classifier: The scikit-learn classifier.
        X_train, X_val: Feature matrices for training and validation.
        y_train, y_val: Labels for training and validation.
    Returns:
        accuracy: Validation accuracy in percentage.
    """
    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict on validation set
    y_pred = classifier.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred) * 100
    return accuracy
