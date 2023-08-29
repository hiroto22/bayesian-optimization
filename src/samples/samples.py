import pandas as pd
import numpy as np
from numpy import matlib

# number_of_generating_samples: 生成するサンプル数
# desired_sum_of_components = 1 合計を指定する特徴量がある場合の、合計の値。例えば、この値を 100 にすれば、合計を 100 にできます
# setting_of_generation_path: 最大値,最小値,グループ番号,有効数字が入ったcsvファイルのpathを指定


# 実験候補の生成
def build_samples(number_of_generating_samples, desired_sum_of_components, setting_of_generation_path):
    setting_of_generation = pd.read_csv(
        setting_of_generation_path, index_col=0, header=0)

    # 0 から 1 の間の一様乱数でサンプル生成
    x_generated = np.random.rand(
        number_of_generating_samples, setting_of_generation.shape[1])

    # 　上限・下限の設定
    x_upper = setting_of_generation.iloc[0, :]  # 上限値
    x_lower = setting_of_generation.iloc[1, :]  # 下限値
    x_generated = x_generated * \
        (x_upper.values - x_lower.values) + x_lower.values  # 上限値から下限値までの間に変換

    # 合計を desired_sum_of_components にする特徴量がある場合
    if setting_of_generation.iloc[2, :].sum() != 0:
        for group_number in range(1, int(setting_of_generation.iloc[2, :].max()) + 1):
            # グループ番号が group_number の特徴量のインデックスを取得
            variable_numbers = np.where(
                setting_of_generation.iloc[2, :] == group_number)[0]

            # 乱数で生成した時の合計を計算 [1.59581414 1.48368493 ...  ]
            actual_sum_of_components = x_generated[:, variable_numbers].sum(
                axis=1)

            # 合計を[[1.59581414 1.59581414 1.59581414] [1.4836 1.4836 1.4836]...]の形に変換
            actual_sum_of_components_converted = np.matlib.repmat(np.reshape(
                actual_sum_of_components, (x_generated.shape[0], 1)), 1, len(variable_numbers))

            x_generated[:, variable_numbers] = x_generated[:, variable_numbers] / \
                actual_sum_of_components_converted * desired_sum_of_components
            deleting_sample_numbers, _ = np.where(x_generated > x_upper.values)
            x_generated = np.delete(
                x_generated, deleting_sample_numbers, axis=0)
            deleting_sample_numbers, _ = np.where(x_generated < x_lower.values)
            x_generated = np.delete(
                x_generated, deleting_sample_numbers, axis=0)

    # 数値の丸め込みをする場合
    if setting_of_generation.shape[0] >= 4:
        for variable_number in range(x_generated.shape[1]):
            x_generated[:, variable_number] = np.round(x_generated[:, variable_number], int(
                setting_of_generation.iloc[3, variable_number]))

    # 保存
    x_generated = pd.DataFrame(
        x_generated, columns=setting_of_generation.columns)
    # 生成したサンプルをcsv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
    x_generated.to_csv('./../data/candidate/generated_samples.csv')


# 実験候補からサンプルを選択する D最適基準
# number_of_selecting_samples 選択するサンプル数
# number_of_random_searches ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数

def select_samples(number_of_selecting_samples, number_of_random_searches, generated_samples_path):
    # D最適基準によってどのサンプルデータを使用するか決める

    x_generated = pd.read_csv(generated_samples_path, index_col=0, header=0)

    autoscaled_x_generated = (
        x_generated - x_generated.mean()) / x_generated.std()

    # 実験条件の候補のインデックスの作成
    all_indexes = list(range(x_generated.shape[0]))

    # D 最適基準に基づくサンプル選択
    np.random.seed(11)  # 乱数を生成するためのシードを固定
    for random_search_number in range(number_of_random_searches):
        # 1. ランダムに候補を選択
        new_selected_indexes = np.random.choice(
            all_indexes, number_of_selecting_samples, replace=False)
        new_selected_samples = autoscaled_x_generated.iloc[new_selected_indexes, :]
        # 2. D 最適基準を計算
        xt_x = np.dot(new_selected_samples.T, new_selected_samples)
        d_optimal_value = np.linalg.det(xt_x)
        # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
        if random_search_number == 0:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_selected_indexes.copy()
        else:
            if best_d_optimal_value < d_optimal_value:
                best_d_optimal_value = d_optimal_value.copy()
                selected_sample_indexes = new_selected_indexes.copy()
    selected_sample_indexes = list(selected_sample_indexes)  # リスト型に変換

    # 選択されたサンプル、選択されなかったサンプル
    # 選択されたサンプル
    selected_samples = x_generated.iloc[selected_sample_indexes, :]
    remaining_indexes = np.delete(
        all_indexes, selected_sample_indexes)  # 選択されなかったサンプルのインデックス
    remaining_samples = x_generated.iloc[remaining_indexes, :]  # 選択されなかったサンプル

    # 保存
    # 選択されたサンプルを csv ファイルに保存。
    selected_samples.to_csv('./../data/candidate/selected_samples.csv')
    # 選択されなかったサンプルを csv ファイルに保存。
    remaining_samples.to_csv('./../data/candidate/remaining_samples.csv')

    print(selected_samples.corr())  # 相関行列の確認
