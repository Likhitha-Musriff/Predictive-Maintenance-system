from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import create_features

def train_model():
    # Load and preprocess data
    df = load_data('data/sensor_data.csv')
    df = create_features(df)  # Feature engineering
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
