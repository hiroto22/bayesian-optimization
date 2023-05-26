import pandas as pd

def build_features(regression_method,x_prediction_data_path, experiment_result_data_path):
    # 選択されていないサンプルの読み込み
    x_prediction = pd.read_csv(x_prediction_data_path, index_col=0, header=0)

    # 実験結果データの読み込み
    dataset = pd.read_csv(experiment_result_data_path, index_col=0, header=0)

    # データ分割
    y = dataset.iloc[:, 0]  # 目的変数
    x = dataset.iloc[:, 1:]  # 説明変数

    # 非線形変換
    if regression_method == 'ols_nonlinear':
        x_tmp = x.copy()
        x_prediction_tmp = x_prediction.copy()
        x_square = x ** 2  # 二乗項
        x_prediction_square = x_prediction ** 2  # 二乗項
        # 二乗項と交差項の追加
        for i in range(x_tmp.shape[1]):
            print(i + 1, '/', x_tmp.shape[1])
            for j in range(x_tmp.shape[1]):
                if i == j:  # 二乗項
                    x = pd.concat([x, x_square.rename(columns={
                                x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]], axis=1)
                    x_prediction = pd.concat([x_prediction, x_prediction_square.rename(columns={
                                            x_prediction_square.columns[i]: '{0}^2'.format(x_prediction_square.columns[i])}).iloc[:, i]], axis=1)
                elif i < j:  # 交差項
                    x_cross = x_tmp.iloc[:, i] * x_tmp.iloc[:, j]
                    x_prediction_cross = x_prediction_tmp.iloc[:,
                                                            i] * x_prediction_tmp.iloc[:, j]
                    x_cross.name = '{0}*{1}'.format(
                        x_tmp.columns[i], x_tmp.columns[j])
                    x_prediction_cross.name = '{0}*{1}'.format(
                        x_prediction_tmp.columns[i], x_prediction_tmp.columns[j])
                    x = pd.concat([x, x_cross], axis=1)
                    x_prediction = pd.concat(
                        [x_prediction, x_prediction_cross], axis=1)


    
    # 標準偏差が 0 の特徴量の削除(データの値が全て同じものを削除)
    deleting_variables = x.columns[x.std() == 0]
    x = x.drop(deleting_variables, axis=1)
    x_prediction.columns = x.columns
    x_prediction = x_prediction.drop(deleting_variables, axis=1)

    # オートスケーリング
    autoscaled_y = (y - y.mean()) / y.std()
    autoscaled_x = (x - x.mean()) / x.std()
    autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

    return x,y,autoscaled_x, autoscaled_y, autoscaled_x_prediction






# # ガウス過程回帰
# def gpr_kernels():
#     regression_method = 'gpr_kernels'
#     elif regression_method == 'gpr_kernels':
#     # クロスバリデーションによるカーネル関数の最適化
#     cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
#     r2cvs = [] # 空の list。主成分の数ごとに、クロスバリデーション後の r2 を入れていきます
#     for index, kernel in enumerate(kernels):
#         print(index + 1, '/', len(kernels))
#         model = GaussianProcessRegressor(alpha=0, kernel=kernel)
#         estimated_y_in_cv = np.ndarray.flatten(cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation))
#         estimated_y_in_cv = estimated_y_in_cv * y.std(ddof=1) + y.mean()
#         r2cvs.append(r2_score(y, estimated_y_in_cv))
#     optimal_kernel_number = np.where(r2cvs == np.max(r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
#     optimal_kernel = kernels[optimal_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数
#     print('クロスバリデーションで選択されたカーネル関数の番号 :', optimal_kernel_number)
#     print('クロスバリデーションで選択されたカーネル関数 :', optimal_kernel)

#     # モデル構築
#     model = GaussianProcessRegressor(alpha=0, kernel=optimal_kernel) # GPR モデルの宣言

#     model.fit(autoscaled_x, autoscaled_y)  # モデル構築
    
#     # 標準回帰係数
#     standard_regression_coefficients = model.coef_
#     standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients, index=x.columns, columns=['standard_regression_coefficients'])
#     standard_regression_coefficients.to_csv(
#         'standard_regression_coefficients_{0}.csv'.format(regression_method))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    
#     # トレーニングデータの推定
#     autoscaled_estimated_y = model.predict(autoscaled_x)  # y の推定
#     estimated_y = autoscaled_estimated_y * y.std() + y.mean()  # スケールをもとに戻す
#     estimated_y = pd.DataFrame(estimated_y, index=x.index, columns=['estimated_y'])

#     # トレーニングデータの実測値 vs. 推定値のプロット
#     plt.rcParams['font.size'] = 18
#     plt.scatter(y, estimated_y.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
#     y_max = max(y.max(), estimated_y.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
#     y_min = min(y.min(), estimated_y.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
#     plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
#             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
#     plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
#     plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
#     plt.xlabel('actual y')  # x 軸の名前
#     plt.ylabel('estimated y')  # y 軸の名前
#     plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
#     plt.show()  # 以上の設定で描画

#     # トレーニングデータのr2, RMSE, MAE
#     print('r^2 for training data :', r2_score(y, estimated_y))
#     print('RMSE for training data :', mean_squared_error(y, estimated_y, squared=False))
#     print('MAE for training data :', mean_absolute_error(y, estimated_y))

#     # トレーニングデータの結果の保存
#     y_for_save = pd.DataFrame(y)
#     y_for_save.columns = ['actual_y']
#     y_error_train = y_for_save.iloc[:, 0] - estimated_y.iloc[:, 0]
#     y_error_train = pd.DataFrame(y_error_train)
#     y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
#     results_train = pd.concat([y_for_save, estimated_y, y_error_train], axis=1) # 結合
#     results_train.to_csv('estimated_y_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

#     # クロスバリデーションによる y の値の推定
#     cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
#     autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)  # y の推定
#     estimated_y_in_cv = autoscaled_estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
#     estimated_y_in_cv = pd.DataFrame(estimated_y_in_cv, index=x.index, columns=['estimated_y'])

#     # クロスバリデーションにおける実測値 vs. 推定値のプロット
#     plt.rcParams['font.size'] = 18
#     plt.scatter(y, estimated_y_in_cv.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
#     y_max = max(y.max(), estimated_y_in_cv.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
#     y_min = min(y.min(), estimated_y_in_cv.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
#     plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
#             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
#     plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
#     plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
#     plt.xlabel('actual y')  # x 軸の名前
#     plt.ylabel('estimated y')  # y 軸の名前
#     plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
#     plt.show()  # 以上の設定で描画

#     # クロスバリデーションにおけるr2, RMSE, MAE
#     print('r^2 in cross-validation :', r2_score(y, estimated_y_in_cv))
#     print('RMSE in cross-validation :', mean_squared_error(y, estimated_y_in_cv, squared=False))
#     print('MAE in cross-validation :', mean_absolute_error(y, estimated_y_in_cv))

#     # クロスバリデーションの結果の保存
#     y_error_in_cv = y_for_save.iloc[:, 0] - estimated_y_in_cv.iloc[:, 0]
#     y_error_in_cv = pd.DataFrame(y_error_in_cv)
#     y_error_in_cv.columns = ['error_of_y(actual_y-estimated_y)']
#     results_in_cv = pd.concat([y_for_save, estimated_y_in_cv, y_error_in_cv], axis=1) # 結合
#     results_in_cv.to_csv('estimated_y_in_cv_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください


