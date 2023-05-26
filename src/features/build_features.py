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



# # サポートベクター回帰(線形カーネル)
# def svr_linear():
#     regression_method = 'svr_linear'
#     regression_method == 'svr_linear':
#     # クロスバリデーションによる C, ε の最適化
#     cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
#     gs_cv = GridSearchCV(SVR(kernel='linear'), {'C':linear_svr_cs, 'epsilon':linear_svr_epsilons}, cv=cross_validation)  # グリッドサーチの設定
#     gs_cv.fit(autoscaled_x, autoscaled_y)  # グリッドサーチ + クロスバリデーション実施
#     optimal_linear_svr_c = gs_cv.best_params_['C']  # 最適な C
#     optimal_linear_svr_epsilon = gs_cv.best_params_['epsilon']  # 最適な ε
#     print('最適化された C : {0} (log(C)={1})'.format(optimal_linear_svr_c, np.log2(optimal_linear_svr_c)))
#     print('最適化された ε : {0} (log(ε)={1})'.format(optimal_linear_svr_epsilon, np.log2(optimal_linear_svr_epsilon)))
#     model = SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon) # SVRモデルの宣言
    
#      # 標準回帰係数
#     if regression_method == 'svr_linear':
#         standard_regression_coefficients = model.coef_.T
    
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


# # サポートベクター回帰(ガウシアンカーネル)
# def svr_gaussian():
#     regression_method = 'svr_gaussian'
#     elif regression_method == 'svr_gaussian':
#     # C, ε, γの最適化
#     # 分散最大化によるガウシアンカーネルのγの最適化
#     variance_of_gram_matrix = []
#     autoscaled_x_array = np.array(autoscaled_x)
#     for nonlinear_svr_gamma in nonlinear_svr_gammas:
#         gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
#         variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
#     optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]

#     cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
#     # CV による ε の最適化
#     r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
#     for nonlinear_svr_epsilon in nonlinear_svr_epsilons:
#         model = SVR(kernel='rbf', C=3, epsilon=nonlinear_svr_epsilon, gamma=optimal_nonlinear_gamma)
#         autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
#         r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
#     optimal_nonlinear_epsilon = nonlinear_svr_epsilons[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

#     # CV による C の最適化
#     r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
#     for nonlinear_svr_c in nonlinear_svr_cs:
#         model = SVR(kernel='rbf', C=nonlinear_svr_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)
#         autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
#         r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
#     optimal_nonlinear_c = nonlinear_svr_cs[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

#     # CV による γ の最適化
#     r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
#     for nonlinear_svr_gamma in nonlinear_svr_gammas:
#         model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=nonlinear_svr_gamma)
#         autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
#         r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
#     optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
#     # 結果の確認
#     print('最適化された C : {0} (log(C)={1})'.format(optimal_nonlinear_c, np.log2(optimal_nonlinear_c)))
#     print('最適化された ε : {0} (log(ε)={1})'.format(optimal_nonlinear_epsilon, np.log2(optimal_nonlinear_epsilon)))
#     print('最適化された γ : {0} (log(γ)={1})'.format(optimal_nonlinear_gamma, np.log2(optimal_nonlinear_gamma)))
#     # モデル構築
#     model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)  # SVR モデルの宣言
    
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


# # ガウス過程回帰
# def gpr_one_kernel():
#     regression_method = 'gpr_one_kernel'  
#     ad_method = 'ocsvm'  # AD設定手法 'knn', 'ocsvm', 'ocsvm_gamma_optimization'

#     fold_number = 10  # クロスバリデーションの fold 数
#     rate_of_training_samples_inside_ad = 0.96  # AD 内となるトレーニングデータの割合。AD　のしきい値を決めるときに使用

#     linear_svr_cs = 2 ** np.arange(-10, 5, dtype=float) # 線形SVR の C の候補
#     linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float) # 線形SVRの ε の候補
#     nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float) # SVR の C の候補
#     nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float) # SVR の ε の候補
#     nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float) # SVR のガウシアンカーネルの γ の候補
#     kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#     k_in_knn = 5  # k-NN における k
#     ocsvm_nu = 0.04  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
#     ocsvm_gamma = 0.1  # OCSVM における γ
#     ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

#     x_prediction = pd.read_csv('remaining_samples.csv', index_col=0, header=0)

#     # データ分割
#     y = dataset.iloc[:, 0]  # 目的変数
#     x = dataset.iloc[:, 1:]  # 説明変数

    
#     # 標準偏差が 0 の特徴量の削除
#     deleting_variables = x.columns[x.std() == 0]
#     x = x.drop(deleting_variables, axis=1)
#     x_prediction = x_prediction.drop(deleting_variables, axis=1)

#     # カーネル 11 種類
#     kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
#             ConstantKernel() * RBF() + WhiteKernel(),
#             ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
#             ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(),
#             ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
#             ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
#             ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
#             ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
#             ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
#             ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
#             ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]

#     # オートスケーリング
#     autoscaled_y = (y - y.mean()) / y.std()
#     autoscaled_x = (x - x.mean()) / x.std()
#     x_prediction.columns = x.columns
#     autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

   
   
    
#     regression_method == 'gpr_one_kernel'
#     selected_kernel = kernels[kernel_number]
#     model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)
   

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

    