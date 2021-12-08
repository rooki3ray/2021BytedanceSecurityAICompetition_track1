# 字节跳动安全AI挑战赛——色情导流用户识别

团队名称：naivenlp
    
## 赛题描述

- [比赛地址](https://security.bytedance.com/fe/ai-challenge#/challenge)
- 输入：用户的特征，包括基础信息、投稿信息、行为信息。
- 输出：用户的标签（1表示色情导流用户，0表示正常用户）
- 评价指标采用$f_{\beta}$（取$\beta=0.3$）
$$
f_{\beta} = (1 + \beta^2)\frac{p*r}{\beta^2*p+r}
$$
### 数据构成

- 用户基础信息
    - 性别、粉丝数、个签、关注人数……
- 用户投稿信息
    - 视频标题、poi、省份、投稿时间
- 用户行为信息
    - 播放次数、点赞数、分享数……

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

按时间排序，saved下最近的一个目录下的csv文件即为测试集的预测结果。# 2021BytedanceSecurityAICompetition_Track1

## 方案说明

- 特征工程
    - log1p 数据平滑
    - 类别特征（LabelEncoder）
    - 时间特征（min-max 归一化）
    - 文本特征（长度、WordVec）
    - 交叉特征
- 模型训练
    - 10折lgb交叉验证，均值作为预测结果
    - 伪标签
- 最终分数线上第二（0.9906）。