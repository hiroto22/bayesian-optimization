import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import matlib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

# 線形重回帰
def ols_linear(autoscaled_x, autoscaled_y,x):
    # モデル構築
    model = LinearRegression()
    model.fit(autoscaled_x, autoscaled_y)  # モデル構築

    standard_regression_coefficients = model.coef_
    standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients, index=x.columns, columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
    './../data/result/standard_regression_coefficients_ols_liner.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    return model


    