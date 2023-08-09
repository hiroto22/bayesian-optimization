import pandas as pd


def build_features(regression_method,x_prediction_data_path, experiment_result_data_path):
    # 選択されていないサンプルの読み込み
    x_prediction = pd.read_csv(x_prediction_data_path, index_col=0, header=0)

    # 実験結果データの読み込み
    dataset = pd.read_csv(experiment_result_data_path, index_col=0, header=0)

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


    
    # 標準偏差が 0 の特徴量の削除(データの値が全て同じカラムを削除)
    deleting_variables = x.columns[x.std() == 0]
    x = x.drop(deleting_variables, axis=1)
    x_prediction.columns = x.columns
    x_prediction = x_prediction.drop(deleting_variables, axis=1)

    # オートスケーリング
    autoscaled_y = (y - y.mean()) / y.std()
    autoscaled_x = (x - x.mean()) / x.std()
    autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

    return x,y,autoscaled_x, autoscaled_y, autoscaled_x_prediction, x_prediction
