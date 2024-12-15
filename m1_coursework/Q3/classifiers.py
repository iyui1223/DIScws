from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

def get_classifiers():
    """
    Returns a dictionary of scikit-learn classifiers.
    """
    return {
        "SVM": SVC(kernel="linear"),
        "Kernel SVM": SVC(kernel="rbf"),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Boosting": AdaBoostClassifier(),
        "Bagging": BaggingClassifier()
    }
