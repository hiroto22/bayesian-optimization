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

# AD
# ad_method: knn ocsvm
def ad(ad_method, autoscaled_x, x, x_prediction, autoscaled_x_prediction, k_in_knn, rate_of_training_samples_inside_ad, estimated_y_prediction, ocsvm_gamma, ocsvm_gammas, ocsvm_nu):
    if ad_method == 'knn':
        ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
        ad_model.fit(autoscaled_x)
        
        # サンプルごとの k 最近傍サンプルとの距離に加えて、k 最近傍サンプルのインデックス番号も一緒に出力されるため、出力用の変数を 2 つに
        # トレーニングデータでは k 最近傍サンプルの中に自分も含まれ、自分との距離の 0 を除いた距離を考える必要があるため、k_in_knn + 1 個と設定
        knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)
        knn_distance_train = pd.DataFrame(knn_distance_train, index=autoscaled_x.index)  # DataFrame型に変換
        mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                                                columns=['mean_of_knn_distance'])  # 自分以外の k_in_knn 個の距離の平均
        mean_of_knn_distance_train.to_csv('mean_of_knn_distance_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
        
        # トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
        sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)  # 距離の平均の小さい順に並び替え
        ad_threshold = sorted_mean_of_knn_distance_train.iloc[
            round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]
        
        # トレーニングデータに対して、AD の中か外かを判定
        inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold
        
        # 予測用データに対する k-NN 距離の計算
        knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_prediction)
        knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_prediction.index)  # DataFrame型に変換
        ad_index_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1), columns=['mean_of_knn_distance'])  # k_in_knn 個の距離の平均
        inside_ad_flag_prediction = ad_index_prediction <= ad_threshold

    elif ad_method == 'ocsvm':
        if ad_method == 'ocsvm_gamma_optimization':
            # 分散最大化によるガウシアンカーネルのγの最適化
            variance_of_gram_matrix = []
            autoscaled_x_array = np.array(autoscaled_x)
            for nonlinear_svr_gamma in ocsvm_gammas:
                gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
                variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
            optimal_gamma = ocsvm_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]
            # 最適化された γ
            print('最適化された gamma :', optimal_gamma)
        else:
            optimal_gamma = ocsvm_gamma
        
        # OCSVM による AD
        ad_model = OneClassSVM(kernel='rbf', gamma=optimal_gamma, nu=ocsvm_nu)  # AD モデルの宣言
        ad_model.fit(autoscaled_x)  # モデル構築

        # トレーニングデータのデータ密度 (f(x) の値)
        data_density_train = ad_model.decision_function(autoscaled_x)
        number_of_support_vectors = len(ad_model.support_)
        number_of_outliers_in_training_data = sum(data_density_train < 0)
        print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
        print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x.shape[0])
        print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
        print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x.shape[0])
        data_density_train = pd.DataFrame(data_density_train, index=x.index, columns=['ocsvm_data_density'])
        data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
        # トレーニングデータに対して、AD の中か外かを判定
        inside_ad_flag_train = data_density_train >= 0
        # 予測用データのデータ密度 (f(x) の値)
        ad_index_prediction = ad_model.decision_function(autoscaled_x_prediction)
        number_of_outliers_in_prediction_data = sum(ad_index_prediction < 0)
        print('\nテストデータにおける外れサンプル数 :', number_of_outliers_in_prediction_data)
        print('テストデータにおける外れサンプルの割合 :', number_of_outliers_in_prediction_data / x_prediction.shape[0])
        ad_index_prediction = pd.DataFrame(ad_index_prediction, index=x_prediction.index, columns=['ocsvm_data_density'])
        ad_index_prediction.to_csv('ocsvm_ad_index_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
        # 予測用トデータに対して、AD の中か外かを判定
        inside_ad_flag_prediction = ad_index_prediction >= 0

    estimated_y_prediction[np.logical_not(inside_ad_flag_prediction)] = -10 ** 10 # AD 外の候補においては負に非常に大きい値を代入し、次の候補として選ばれないようにします

    return inside_ad_flag_prediction, estimated_y_prediction
