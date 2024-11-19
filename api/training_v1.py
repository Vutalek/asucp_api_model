import math
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import model_selection
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn import metrics
import dill as pickle

data_path = "./data/data_v1.xlsx"
model_path = "./models/model_v1.pk"

def train():
    data = pd.read_excel(data_path, parse_dates = ["Дата"], date_format = "%d.%m.%Y", engine = "openpyxl")

    X = data.copy()
    X = X.drop("Количество товара", axis = 1)

    y = data["Количество товара"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 777)

    pipe = make_pipeline(
        PreProcessor(),
        TransformedTargetRegressor(regressor = QuantileRegressor(), transformer = Ceil())
        )
    
    param_grid = {
        "transformedtargetregressor__regressor__quantile": [0.3, 0.4, 0.5, 0.6, 0.7],
        "transformedtargetregressor__regressor__alpha": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    grid = model_selection.GridSearchCV(pipe, param_grid = param_grid, scoring = "r2")
    grid.fit(X_train, y_train) 

    percent = metrics.r2_score(y_test, grid.predict(X_test)) * 100

    return (grid, percent)
    

    
class PreProcessor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, df, y, **fit_params):
        self.dataset = df.copy()
        self.dataset["Количество товара"] = y
        return self
        
    def transform(self, df):
        
        df["month-sin"] = df["Дата"].dt.month
        df["month-sin"] = df["month-sin"] * (2 * math.pi / 12)
        df["month-sin"] = df["month-sin"].apply(math.sin)

        df["month-cos"] = df["Дата"].dt.month
        df["month-cos"] = df["month-cos"] * (2 * math.pi / 12)
        df["month-cos"] = df["month-cos"].apply(math.cos)

        df["first-third"] = df["Дата"].dt.day.apply(lambda day: (0, 1)[day >= 0 and day <= 10])
        df["second-third"] = df["Дата"].dt.day.apply(lambda day: (0, 1)[day >= 11 and day <= 20])
        df["third-third"] = df["Дата"].dt.day.apply(lambda day: (0, 1)[day >= 21 and day <= 31])

        df["day-sin"] = df["Дата"].dt.day_of_week + 1
        df["day-sin"] = df["day-sin"] * (2 * math.pi / 7)
        df["day-sin"] = df["day-sin"].apply(math.sin)

        df["day-cos"] = df["Дата"].dt.day_of_week + 1
        df["day-cos"] = df["day-cos"] * (2 * math.pi / 7)
        df["day-cos"] = df["day-cos"].apply(math.cos)

        df["2k-mean-product"] = df.apply(lambda row: self.two_mean_product(row), axis = 1)

        df = df.drop(["Дата", "Товар"], axis = 1)

        df = df.rename(columns = {"Склад": "N_warehouse"})

        return df

    def two_mean_product(self, row):
        differ = pd.DataFrame()
        differ = self.dataset[(self.dataset["Товар"] == row["Товар"]) & (self.dataset["Склад"] == row["Склад"])]
        differ["Дата"] = differ["Дата"] - row["Дата"]
        past = differ[differ["Дата"].dt.days < 0]
        future = differ[differ["Дата"].dt.days > 0]
        past = past.sort_values(by = "Дата", ascending = False)
        future = future.sort_values(by = "Дата")
        if future.empty and past.empty:
            return 0
        elif future.empty:
            return past["Количество товара"].iloc[0]
        elif past.empty:
            return future["Количество товара"].iloc[0]
        else:
            return (future["Количество товара"].iloc[0] + past["Количество товара"].iloc[0])/2
        
class Ceil(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
        
    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return np.ceil(y)
        
if __name__ == "__main__":
    print("Start training.")
    model, percent = train()
    print("Training complete.")

    print(f"Best parameters: {model.best_params_}")

    print(f"R2 score(%): {percent}%")

    print("Serializing model.")
    with open(model_path, "wb") as file:
        pickle.dump(model, file, recurse = True)
    print("Serializing complete.")
