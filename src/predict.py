import joblib
import pandas as pd

def load_model(path="model/model.pkl"):
    return joblib.load(path)

def predict(model, input_data):
    return model.predict(input_data)

if __name__ == "__main__":
    model = load_model()
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = predict(model, sample)
    print(f"Prediction: {prediction[0]}")