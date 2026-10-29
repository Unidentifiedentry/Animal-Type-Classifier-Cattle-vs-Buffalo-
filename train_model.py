# requirements:
# pip install opencv-python scikit-learn matplotlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === CONFIG ===
IMG_SIZE = (128, 128)

# === TRAIN MODEL ===
def train_classifier(X, y, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    return clf

# Example usage:
# X, y = ... # Load your data and labels
# train_classifier(X, y, 'buffalo_classifier.joblib')