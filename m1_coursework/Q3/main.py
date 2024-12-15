from read_data import prepare_data
from classifiers import get_classifiers
from train_evaluate import train_and_evaluate

def main():
    # Prepare the data
    X_train, X_val, y_train, y_val = prepare_data()

    # Get classifiers
    classifiers = get_classifiers()

    # Train and evaluate each classifier
    results = {}
    for name, clf in classifiers.items():
        accuracy = train_and_evaluate(clf, X_train, X_val, y_train, y_val)
        results[name] = accuracy
        print(f"{name}: Validation Accuracy = {accuracy:.2f}%")

    # Print final results
    print("\nSummary of Results:")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")

if __name__ == "__main__":
    main()
