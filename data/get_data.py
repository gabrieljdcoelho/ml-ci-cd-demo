from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame
df["species"] = data.target_names[data.target]
df.drop(columns=["target"], inplace=True)
df.to_csv("data/iris.csv", index=False)