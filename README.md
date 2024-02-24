baysian-optimization
==============================

Using baysian-optimazation for cell free 



<h2>ディレクトリ構成</h2>

------------

    ├── .vscode
    ├── analysis                       　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　<- データ分析用のコード
    ├── bayesian_optimization          　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　<- 次の実験候補を選択するために使われるコード。初回候補をD最適基準で選ぶためのコードもここにある
    ├── build_samples                                              <- 実験候補を作成するためのコード
    ├── data
    │   ├── acuisiton_function_prediction                          <- 獲得関数の計算結果
    │   ├── estimated_y_prediction_gpr_one_kernel                  <- 発現量の予測値
    │   ├── estimated_y_prediction_gpr_one_kernel_std              <- 発現量の予測値の標準偏差
    │　　  ├── old_data                                               <- 最初の頃使用していたが失敗したため使っていないデータ
    │   ├── remaining_samples                                      <- 実験ごとに未使用の実験候補を保存する
    │   ├── result                                                 <- 毎回の実験結果を保存する
    │   ├── next_samples                                           <- 選択された次の実験候補
    │   ├── generated_samples.csv                                  <- build_samplesで作成された実験候補
    │   └──  result.csv                                            <- 全ラウンドの実験結果を一つのファイルにまとめたもの   
    │
    ├── env 
    ├── .env
    ├── .gitignore
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── test_environment.py
    └── tox.ini            


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
