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

# サポートベクター回帰(ガウシアンカーネル)
def svr_gaussian(fold_number,autoscaled_x, autoscaled_y,y):
    nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # SVR の C の候補
    nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # SVR の ε の候補
    nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # SVR のガウシアンカーネルの γ の候補

    # C, ε, γの最適化
    # 分散最大化によるガウシアンカーネルのγの最適化
    variance_of_gram_matrix = []
    autoscaled_x_array = np.array(autoscaled_x)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]

    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    # CV による ε の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for nonlinear_svr_epsilon in nonlinear_svr_epsilons:
        model = SVR(kernel='rbf', C=3, epsilon=nonlinear_svr_epsilon, gamma=optimal_nonlinear_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_nonlinear_epsilon = nonlinear_svr_epsilons[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

    # CV による C の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for nonlinear_svr_c in nonlinear_svr_cs:
        model = SVR(kernel='rbf', C=nonlinear_svr_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_nonlinear_c = nonlinear_svr_cs[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

    # CV による γ の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=nonlinear_svr_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
    # 結果の確認
    print('最適化された C : {0} (log(C)={1})'.format(optimal_nonlinear_c, np.log2(optimal_nonlinear_c)))
    print('最適化された ε : {0} (log(ε)={1})'.format(optimal_nonlinear_epsilon, np.log2(optimal_nonlinear_epsilon)))
    print('最適化された γ : {0} (log(γ)={1})'.format(optimal_nonlinear_gamma, np.log2(optimal_nonlinear_gamma)))
    # モデル構築
    model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)  # SVR モデルの宣言
    model.fit(autoscaled_x, autoscaled_y)

    return model


# ガウス過程回帰
def gpr_one_kernel(kernel_number,autoscaled_x, autoscaled_y,x):
    # カーネル 11 種類
    kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
            ConstantKernel() * RBF() + WhiteKernel(),
            ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(),
            ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]

    selected_kernel = kernels[kernel_number]
    model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)
    model.fit(autoscaled_x, autoscaled_y)

    return model