# 字节跳动安全AI挑战赛——色情导流用户识别

团队名称：naivenlp

## 测试环境

- Ubuntu 16.04.6 LTS
- Intel(R) Xeon(R) CPU E5-2640 v3 @ 2.60GHz

## 安装依赖

python依赖如下

```
lightgbm==3.2.1
numpy==1.19.2
pandas==1.1.5
sklearn==0.0
gensim==4.1.2
tqdm==4.50.2
```

安装依赖

```sh
pip install -r requirements.txt
```

## 使用方法

直接运行run.sh脚本即可

```sh
chmod +x run.sh
./run.sh
```

会在当前目录创建saved目录，目录结构如下

```
.
├── 1_word2vec.py
├── 2_merge_data.py
├── 3_5_train_kfold.py
├── 4_pseudo_label.py
├── config.py
├── data
│   ├── pseudo.csv
│   ├── raw
│   │   ├── 测试数据
│   │   └── 训练数据
│   ├── sentence
│   │   └── signature
│   ├── test.csv
│   ├── train.csv
│   └── ...
├── evaluate_kfold.py
├── __pycache__
├── readme.md
├── requirements.txt
├── run.sh
├── saved
│   ├── 1112_1315_0.985_0.9934
│   │   └── ...
│   ├── 1112_1320_0.985_pseudo_0.9934
│   │   └── ...
│   ├── 1112_1321_pseudo_0.985_0.9942
│   │   ├── 1112_1321_0.985_results_kfold_0.9942.csv
│   │   ├── log.log
│   │   └── ...
└── utils.py
```

总耗时约15分钟，请耐心等待。

按时间排序，saved下最近的一个目录下的csv文件即为测试集的预测结果。# 2021BytedanceSecurityAICompetition_Stage1
