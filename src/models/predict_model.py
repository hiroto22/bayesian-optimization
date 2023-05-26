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

def predict(regression_method,fold_number, autoscaled_x, autoscaled_y, autoscaled_x_prediction,x, y, model):
     # クロスバリデーションの fold 数
    fold_number = fold_number
    
    # トレーニングデータの推定
    autoscaled_estimated_y = model.predict(autoscaled_x)  # y の推定
    estimated_y = autoscaled_estimated_y * y.std() + y.mean()  # スケールをもとに戻す
    estimated_y = pd.DataFrame(estimated_y, index=x.index, columns=['estimated_y'])

    # トレーニングデータの実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18
    plt.scatter(y, estimated_y.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y.max(), estimated_y.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y.min(), estimated_y.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
    plt.show()  # 以上の設定で描画

    # トレーニングデータのr2, RMSE, MAE
    print('r^2 for training data :', r2_score(y, estimated_y))
    print('RMSE for training data :', mean_squared_error(y, estimated_y, squared=False))
    print('MAE for training data :', mean_absolute_error(y, estimated_y))

    # トレーニングデータの結果の保存
    y_for_save = pd.DataFrame(y)
    y_for_save.columns = ['actual_y']
    y_error_train = y_for_save.iloc[:, 0] - estimated_y.iloc[:, 0]
    y_error_train = pd.DataFrame(y_error_train)
    y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
    results_train = pd.concat([y_for_save, estimated_y, y_error_train], axis=1) # 結合
    results_train.to_csv('./../data/result/estimated_y_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    
    # クロスバリデーションによる y の値の推定
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)  # y の推定
    estimated_y_in_cv = autoscaled_estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
    estimated_y_in_cv = pd.DataFrame(estimated_y_in_cv, index=x.index, columns=['estimated_y'])

    # クロスバリデーションにおける実測値 vs. 推定値のプロット
    plt.rcParams['font.size'] = 18
    plt.scatter(y, estimated_y_in_cv.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
    y_max = max(y.max(), estimated_y_in_cv.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
    y_min = min(y.min(), estimated_y_in_cv.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
    plt.xlabel('actual y')  # x 軸の名前
    plt.ylabel('estimated y')  # y 軸の名前
    plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
    plt.show()  # 以上の設定で描画

    # クロスバリデーションにおけるr2, RMSE, MAE
    print('r^2 in cross-validation :', r2_score(y, estimated_y_in_cv))
    print('RMSE in cross-validation :', mean_squared_error(y, estimated_y_in_cv, squared=False))
    print('MAE in cross-validation :', mean_absolute_error(y, estimated_y_in_cv))

    # クロスバリデーションの結果の保存
    y_error_in_cv = y_for_save.iloc[:, 0] - estimated_y_in_cv.iloc[:, 0]
    y_error_in_cv = pd.DataFrame(y_error_in_cv)
    y_error_in_cv.columns = ['error_of_y(actual_y-estimated_y)']
    results_in_cv = pd.concat([y_for_save, estimated_y_in_cv, y_error_in_cv], axis=1) # 結合
    results_in_cv.to_csv('./../data/result/estimated_y_in_cv_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください