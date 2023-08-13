from scipy.stats import norm
import numpy as np
import pandas as pd


def acquisition_function(x_prediction, estimated_y_prediction, estimated_y_prediction_std, delta, acquisition_function, relaxation, y, regression_method):
    target_range = [0, 1]  # PTR
    relaxation = 0.01  # EI, PI
    delta = 10 ** -6  # MI

    cumulative_variance = np.zeros(
        x_prediction.shape[0])  # MI で必要な "ばらつき" を 0 で初期化
    if acquisition_function == 'MI':
        acquisition_function_prediction = estimated_y_prediction + np.log(2 / delta) ** 0.5 * (
            (estimated_y_prediction_std ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
        cumulative_variance = cumulative_variance + estimated_y_prediction_std ** 2
    elif acquisition_function == 'EI':
        acquisition_function_prediction = (estimated_y_prediction - max(y) - relaxation * y.std()) * \
            norm.cdf((estimated_y_prediction - max(y) - relaxation * y.std()) /
                     estimated_y_prediction_std) + \
            estimated_y_prediction_std * \
            norm.pdf((estimated_y_prediction - max(y) - relaxation * y.std()) /
                     estimated_y_prediction_std)
    elif acquisition_function == 'PI':
        acquisition_function_prediction = norm.cdf(
            (estimated_y_prediction - max(y) - relaxation * y.std()) / estimated_y_prediction_std)
    elif acquisition_function == 'PTR':
        acquisition_function_prediction = norm.cdf(target_range[1],
                                                   loc=estimated_y_prediction,
                                                   scale=estimated_y_prediction_std
                                                   ) - norm.cdf(target_range[0],
                                                                loc=estimated_y_prediction,
                                                                scale=estimated_y_prediction_std)
    acquisition_function_prediction[estimated_y_prediction_std <= 0] = 0

    # 保存
    estimated_y_prediction = pd.DataFrame(
        estimated_y_prediction, x_prediction.index, columns=['estimated_y'])
    estimated_y_prediction.to_csv('estimated_y_prediction_{0}.csv'.format(
        regression_method))  # 予測結果を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    estimated_y_prediction_std = pd.DataFrame(
        estimated_y_prediction_std, x_prediction.index, columns=['std_of_estimated_y'])
    estimated_y_prediction_std.to_csv('estimated_y_prediction_{0}_std.csv'.format(
        regression_method))  # 予測値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    acquisition_function_prediction = pd.DataFrame(
        acquisition_function_prediction, index=x_prediction.index, columns=['acquisition_function'])
    acquisition_function_prediction.to_csv('acquisition_function_prediction_{0}_{1}.csv'.format(
        regression_method, acquisition_function))  # 獲得関数を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

    return estimated_y_prediction, acquisition_function_prediction, acquisition_function
