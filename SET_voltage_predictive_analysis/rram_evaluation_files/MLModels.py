from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np


class mlModels:
    def __init__(self):
        # Create linear regression object
        self.rfr = RandomForestRegressor()
        self.ada = AdaBoostRegressor()

    def runModels(self, X, y, grid_search=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42, shuffle=True)

        # Parameter grid for RandomForestRegressor
        rfr_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Parameter grid for AdaBoostRegressor
        ada_param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }

        if grid_search:
            # GridSearchCV for RandomForestRegressor
            rfr_grid_search = GridSearchCV(estimator=self.rfr, param_grid=rfr_param_grid, cv=5, n_jobs=-1, verbose=1)
            rfr_grid_search.fit(X_train, y_train)
            self.rfr = rfr_grid_search.best_estimator_

            # GridSearchCV for AdaBoostRegressor
            ada_grid_search = GridSearchCV(estimator=self.ada, param_grid=ada_param_grid, cv=5, n_jobs=-1, verbose=1)
            ada_grid_search.fit(X_train, y_train)
            self.ada = ada_grid_search.best_estimator_
            
       
            
        else:
            # Use the default settings
            self.rfr = RandomForestRegressor()
            self.ada = AdaBoostRegressor()
            self.rfr.fit(X_train,y_train)
            self.ada.fit(X_train,y_train)

        # Make predictions using the testing set
        y_pred_rfr = self.rfr.predict(X_test)
        y_pred_ada = self.ada.predict(X_test)


        # For Random Forest Regression
        rfr_metrics = {
            "MSE": mean_squared_error(y_test, y_pred_rfr),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rfr)),
            "MAE": mean_absolute_error(y_test, y_pred_rfr),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_rfr),
            "R2": r2_score(y_test, y_pred_rfr),
            "Explained Variance": explained_variance_score(y_test, y_pred_rfr),
            
        }

        # For Adaboost Regression
        ada_metrics = {
            "MSE": mean_squared_error(y_test, y_pred_ada),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ada)),
            "MAE": mean_absolute_error(y_test, y_pred_ada),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_ada),
            "R2": r2_score(y_test, y_pred_ada),
            "Explained Variance": explained_variance_score(y_test, y_pred_ada),
            
        }

        # Return the metrics
        return {
            'rfr_metrics': rfr_metrics,
            'ada_metrics': ada_metrics
        }

    def newSample(self, X, model_name):
        if model_name == 'ada':
            y_pred_ada = self.ada.predict(X)
            #print("Ada Boost: ", y_pred_ada)
            return y_pred_ada
        if model_name == 'rfr':
            y_pred_rfr = self.rfr.predict(X)
            #print("Random Forest: ", y_pred_rfr)
            return y_pred_rfr
