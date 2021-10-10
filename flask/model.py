import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tsa.arima.model import ARIMA
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler


class BudgetModel():
    def __init__(self, X, y, years_to_predict=2, random_seed=42):
        self.SEED = random_seed
        self.years_to_predict = years_to_predict
        self.X_train, self.X_test = 0, 0
        self.y_train, self.y_test = 0, 0
        self.feature_importance = []

        self.X = X
        self.y = y

        self.X = self.X.dropna(axis=1)
        self.X_train, self.X_test = self.X[:-self.years_to_predict], self.X[-self.years_to_predict:]
        self.y_train, self.y_test = self.y[:-self.years_to_predict], self.y[-self.years_to_predict:]

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def predict(self):
        catboost_params = {
            'depth': 1,
            'iterations': 10000,
            'learning_rate': 0.5,
            'l2_leaf_reg': 200,
            'eval_metric': 'MAE',
            'od_type': 'IncToDec',
            'od_pval': 2,
            'silent': True,
            'thread_count': -1,
            'random_seed': self.SEED
        }

        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge()
        }

        dfs = []

        for target in self.y.columns[0:]:
            df_predict = pd.DataFrame({'model': [], 'predicted': [], 'predicted_val': []})

            for k in models:
                models[k].fit(self.X_train, self.y_train[target])
                predicted = models[k].predict(self.X_test)
                predicted_val = models[k].predict(self.X_train)
                df_predict = df_predict.append({'model': k,
                                                'predicted': predicted,
                                                'predicted_val': predicted_val},
                                               ignore_index=True)

            model = CatBoostRegressor(**catboost_params)
            model.fit(self.X_train, self.y_train[target])
            predicted = model.predict(self.X_test)
            predicted_val = model.predict(self.X_train)
            self.feature_importance.append(pd.DataFrame(model.feature_importances_, index=self.X.columns))
            df_predict = df_predict.append({'model': 'CatBoost',
                                            'predicted': predicted,
                                            'predicted_val': predicted_val},
                                           ignore_index=True)

            model = ARIMA(self.y_train[target], order=(3, 2, 1))
            arima_result = model.fit()
            arima_pred = arima_result.predict(start=0, end=len(self.y) - 1)
            df_predict = df_predict.append({'model': 'ARIMA',
                                            'predicted': arima_pred.values[-self.years_to_predict:],
                                            'predicted_val': arima_pred.values[:-self.years_to_predict]},
                                           ignore_index=True)

            df_stacking_test = pd.DataFrame(
                {y: predict.tolist() for (y, predict) in zip(df_predict['model'], df_predict['predicted'])})
            df_stacking_train = pd.DataFrame(
                {y: predict.tolist() for (y, predict) in zip(df_predict['model'], df_predict['predicted_val'])})

            model = Ridge()
            model.fit(df_stacking_train, self.y_train[target])
            predicted = model.predict(df_stacking_test)

            dfs.append(predicted)

        return dfs

    @property
    def feature_importances_(self):
        return self.feature_importance
