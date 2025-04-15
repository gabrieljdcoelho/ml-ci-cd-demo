import pandas as pd
from src.train import train_model

def test_model_training():
    X = pd.DataFrame({
        "sepal_length": [5.1, 4.9],
        "sepal_width": [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width": [0.2, 0.2]
    })
    y = ["setosa", "setosa"]
    model = train_model(X, y)
    assert hasattr(model, "predict")