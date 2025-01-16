import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # Load dataset
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Feature and target split
    X = df.drop(columns=['failure', 'timestamp'])
    y = df['failure']

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y):
    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)
