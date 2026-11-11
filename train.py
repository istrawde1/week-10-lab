import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

 
df = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

X = df[["100g_USD"]]
y = df["rating"]

model = LinearRegression()
model.fit(X, y)

with open("model_1.pickle", "wb") as f:
    pickle.dump(model, f)


roast_codes = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4
}


def roast_category(roast):
    """Return the numeric code for a roast label."""
    return roast_codes.get(roast)

df["roast_cat"] = df["roast"].map(roast_category)

features = ["100g_USD", "roast_cat"]

X = df[features]
y = df["rating"]

dtr = DecisionTreeRegressor()
dtr.fit(X, y)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)

