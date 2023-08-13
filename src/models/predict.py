import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors

# 予測


def predict(regression_method, model, autoscaled_x_prediction, x_prediction, y):
    estimated_y_prediction_std = None

    if regression_method == 'gpr_one_kernel' or regression_method == 'gpr_kernels':  # 標準偏差あり
        estimated_y_prediction, estimated_y_prediction_std = model.predict(
            autoscaled_x_prediction, return_std=True)
        estimated_y_prediction_std = estimated_y_prediction_std * y.std()
        estimated_y_prediction_std = pd.DataFrame(
            estimated_y_prediction_std, x_prediction.index, columns=['std_of_estimated_y'])
        estimated_y_prediction_std.to_csv('estimated_y_prediction_{0}_std.csv'.format(
            regression_method))  # 予測値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    else:
        estimated_y_prediction = model.predict(autoscaled_x_prediction)

    estimated_y_prediction = estimated_y_prediction * y.std() + y.mean()
    estimated_y_prediction = pd.DataFrame(
        estimated_y_prediction, x_prediction.index, columns=['estimated_y'])
    estimated_y_prediction.to_csv('estimated_y_prediction_{0}.csv'.format(
        regression_method))  # 予測結果を csv ファイルに保存。

    return estimated_y_prediction, estimated_y_prediction_std
