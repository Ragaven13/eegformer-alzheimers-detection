from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/checkpoints")

LABEL_NAMES = ["Alzheimer's", "Healthy Control", "FTD"]


def load_data():
    X = np.load(PROCESSED_DIR / "X_windows.npy")
    y = np.load(PROCESSED_DIR / "y_labels.npy")
    subject_ids = np.load(PROCESSED_DIR / "subject_ids.npy", allow_pickle=True)

    train_subjects = np.load(PROCESSED_DIR / "train_subjects.npy", allow_pickle=True)
    val_subjects = np.load(PROCESSED_DIR / "val_subjects.npy", allow_pickle=True)
    test_subjects = np.load(PROCESSED_DIR / "test_subjects.npy", allow_pickle=True)

    return X, y, subject_ids, train_subjects, val_subjects, test_subjects


def extract_statistical_features(X):
    mean_features = X.mean(axis=2)
    std_features = X.std(axis=2)
    min_features = X.min(axis=2)
    max_features = X.max(axis=2)
    power_features = np.mean(X ** 2, axis=2)

    features = np.concatenate(
        [
            mean_features,
            std_features,
            min_features,
            max_features,
            power_features,
        ],
        axis=1,
    )

    return features


def split_by_subject(X_features, y, subject_ids, split_subjects):
    mask = np.isin(subject_ids, split_subjects)

    X_split = X_features[mask]
    y_split = y[mask]

    return X_split, y_split


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)

    X, y, subject_ids, train_subjects, val_subjects, test_subjects = load_data()

    print("\nRaw window shape:")
    print(X.shape)

    X_features = extract_statistical_features(X)

    print("\nFeature shape:")
    print(X_features.shape)
    print("Meaning: windows × statistical features")

    X_train, y_train = split_by_subject(
        X_features,
        y,
        subject_ids,
        train_subjects,
    )

    X_val, y_val = split_by_subject(
        X_features,
        y,
        subject_ids,
        val_subjects,
    )

    X_test, y_test = split_by_subject(
        X_features,
        y,
        subject_ids,
        test_subjects,
    )

    print("\nTrain shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("\nEvaluating on validation set...")
    val_preds = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")

    print("\nValidation Results")
    print("-" * 60)
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Macro F1: {val_f1:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_preds, target_names=LABEL_NAMES))

    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    print("\nEvaluating on test set...")
    test_preds = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    print("\nTest Results")
    print("-" * 60)
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_f1:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=LABEL_NAMES))

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    model_path = RESULTS_DIR / "random_forest_baseline.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model to: {model_path}")


if __name__ == "__main__":
    main()