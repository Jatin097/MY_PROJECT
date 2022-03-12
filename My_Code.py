import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import mlflow
from math import sqrt
import warnings
import mlflow.pyfunc
import sys

diabetes = datasets.load_diabetes()
diabetespd = pd.DataFrame(data=diabetes.data)
diabetespd.to_csv('diabetes.txt', encoding='utf-8', index=False)
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X[0:5]

mlflow.sklearn.autolog()
with mlflow.start_run():
  # 1st idea
  #diabetes_X = diabetes.data[:, np.newaxis, 2]
  
  # 2nd idea
  diabetes_X = diabetes.data
  
  diabetes_X_train = diabetes_X[:-20]
  diabetes_X_test = diabetes_X[-20:]

  diabetes_y_train = diabetes.target[:-20]
  diabetes_y_test = diabetes.target[-20:]

  regr = linear_model.LinearRegression()

  #regr = linear_model.Lasso(alpha=0.1)
 # mlflow.log_param("alpha", 0.1)
  
  regr = linear_model.LassoLars(alpha=0.1)
  mlflow.log_param("alpha", 0.1)

  #regr = linear_model.BayesianRidge()   

  regr.fit(diabetes_X_train, diabetes_y_train)

  diabetes_y_pred = regr.predict(diabetes_X_test)

  mlflow.log_metric("mse", mean_squared_error(diabetes_y_test, diabetes_y_pred))
  mlflow.log_metric("rmse", sqrt(mean_squared_error(diabetes_y_test, diabetes_y_pred)))


  mlflow.log_artifact("diabetes.txt")

def mlflow_run(params, run_name="Diabates"):
  with mlflow.start_run(run_name=run_name) as run:
    # get current run and experiment id
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
 return (experimentID, runID)
if __name__ == '__main__':
   # suppress any deprecated warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)

parameters = [{'alpha': 0.1}]
   (exp_id, run_id) = mlflow_run(params)

   print(f"Finished Experiment id={exp_id} and run id = {run_id}")
