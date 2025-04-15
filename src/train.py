import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

if __name__ == "__main__":
    data = load_data("data/iris.csv")
    X = data.drop("species", axis=1)
    y = data["species"]

    model = train_model(X, y)
    save_model(model, "model/model.pkl")