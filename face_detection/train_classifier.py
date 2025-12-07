# train_classifier.py
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import joblib

EMBED_DIR = "face_embeddings"
MODEL_DIR = "face_model"

def main():
    # Load embeddings and labels
    X = np.load(os.path.join(EMBED_DIR, "X.npy"))
    y = joblib.load(os.path.join(EMBED_DIR, "y.joblib"))

    print("Total samples loaded:", len(y))

    # Count classes before filtering
    counts = Counter(y)
    print("Original class counts:", counts)

    # Filter out classes with fewer than 2 samples
    valid_labels = [label for label, c in counts.items() if c >= 2]

    X_filtered = []
    y_filtered = []

    for emb, label in zip(X, y):
        if label in valid_labels:
            X_filtered.append(emb)
            y_filtered.append(label)

    X_filtered = np.array(X_filtered)
    y_filtered = np.array(y_filtered)

    print(f"Filtered samples: {len(y_filtered)}")
    print(f"Removed classes with <2 samples: {set(counts.keys()) - set(valid_labels)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_filtered)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42
    )

    # Train SVM
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    unique_labels = np.unique(y_test)  # all labels actually in test set
    target_names = le.inverse_transform(unique_labels)
    # Use only filtered classes for target_names
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))


    # Save model + encoder
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "svm_face_recognizer.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))



if __name__ == "__main__":
    main()
