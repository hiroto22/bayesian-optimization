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

    # 標準回帰係数
    standard_regression_coefficients = model.coef_
    standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients, index=x.columns, columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
    './../data/result/standard_regression_coefficients_ols_linear.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    return model


# 非線形重回帰
def ols_nonlinear(autoscaled_x, autoscaled_y,x):
    # モデル構築
    model = LinearRegression()
    model.fit(autoscaled_x, autoscaled_y)  # モデル構築

    # 標準回帰係数
    standard_regression_coefficients = model.coef_
    standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients, index=x.columns, columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
    './../data/result/standard_regression_coefficients_ols_nonlinear.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    return model

# サポートベクター回帰(線形カーネル)
def svr_linear(fold_number,autoscaled_x, autoscaled_y,x):
    linear_svr_cs = 2 ** np.arange(-10, 5, dtype=float)  # 線形SVR の C の候補
    linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # 線形SVRの ε の候補

    # クロスバリデーションによる C, ε の最適化
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    gs_cv = GridSearchCV(SVR(kernel='linear'), {'C':linear_svr_cs, 'epsilon':linear_svr_epsilons}, cv=cross_validation)  # グリッドサーチの設定
    gs_cv.fit(autoscaled_x, autoscaled_y)  # グリッドサーチ + クロスバリデーション実施
    optimal_linear_svr_c = gs_cv.best_params_['C']  # 最適な C
    optimal_linear_svr_epsilon = gs_cv.best_params_['epsilon']  # 最適な ε
    print('最適化された C : {0} (log(C)={1})'.format(optimal_linear_svr_c, np.log2(optimal_linear_svr_c)))
    print('最適化された ε : {0} (log(ε)={1})'.format(optimal_linear_svr_epsilon, np.log2(optimal_linear_svr_epsilon)))
    model = SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon) # SVRモデルの宣言
    model.fit(autoscaled_x, autoscaled_y)  # モデル構築
    
    # 標準回帰係数
    standard_regression_coefficients = model.coef_.T
    standard_regression_coefficients = pd.DataFrame(
        standard_regression_coefficients, index=x.columns, columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
        './../data/result/standard_regression_coefficients_svr_liner.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    return model

    