import logging
import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import Config
from utils import (get_pseudo_data, get_train_and_test_data, load_model,
                   process_feature, evaluate)

config = Config()
train_data, test_data = get_train_and_test_data(config)
rt = test_data.copy()

train_x = train_data.drop(config.drop_list, axis=1)  # 训练集输入
train_y = train_data['label']  # 训练集标签

test_x = test_data.drop(config.drop_list, axis=1)  # 测试集输入
if config.use_pseudo:
    pseudo_x, pseudo_y = get_pseudo_data(config)
    train_x, test_x = process_feature(train_x,
                                      test_x,
                                      config,
                                      pseudo_data=pseudo_x)
    train_y = pd.concat([train_y, pseudo_y], ignore_index=True)
else:
    train_x, test_x = process_feature(train_x, test_x, config)
train_x.to_csv("./data/train_processed.csv", index=False)
train_y.to_csv("./data/train_label.csv", index=False)
test_x.to_csv("./data/test_processed.csv", index=False)
config.logger.info(train_y.value_counts())
config.logger.info(f"Features: {train_x.columns}")
config.logger.info(
    f"Train: {train_x.shape} {train_y.shape} Test: {test_x.shape}")

# 十折交叉验证
folds = StratifiedKFold(n_splits=config.num_fold,
                        shuffle=True,
                        random_state=3123)
predictions = np.zeros(len(test_data))

fbeta_score = 0.0
p_score = r_score = f1_score = 0.0
test_predictions = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    config.logger.info("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], train_y[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx],
                           train_y[val_idx],
                           free_raw_data=False)
    clf = load_model(config, config.lgb_save_path + str(fold_))
    # clf = load_model(config, './saved/1015_1603_0.8_0.9277/lgb.bin' + str(fold_))
    probs = clf.predict(test_x, num_iteration=clf.best_iteration)
    if config.use_vote:
        vote_predictions = (probs >= config.test_threshold).astype('int')
        test_predictions.append(vote_predictions)
    else:
        predictions += probs / folds.n_splits
    fbeta, p, r, f1 = evaluate(config, clf, val_data, "dev")
    fbeta_score += fbeta / folds.n_splits
    p_score += p / folds.n_splits
    r_score += r / folds.n_splits
    f1_score += f1 / folds.n_splits
config.logger.info(f"Current fbeta: {fbeta_score:.4f} p: {p_score:.4f} r: {r_score:.4f} f1: {f1_score:.4f}")

rt['label'] = (predictions >= 0.999).astype('int')
results = rt[rt['label'] == 1]
print(results.shape)
os.system(f"mv {config.cur_dir} {config.cur_dir}_pseudo_{fbeta_score:.4f}")
rst_path = config.result_path.replace(config.cur_dir, f"{config.cur_dir}_pseudo_{fbeta_score:.4f}").replace('results', f'results_pseudo_{fbeta_score:.4f}')
results.to_csv(rst_path,
          index=False,
          sep=',')

os.system(f"cp {rst_path} {config.pseudo_path}")