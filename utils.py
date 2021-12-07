import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from config import Config
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder


def extract_basic_time_(df, col_name):
    # 1. 年
    df[col_name+'year'] = df[col_name].dt.year
    # 2. 月
    df[col_name+'month'] = df[col_name].dt.month
    # 3. 日
    # df[col_name+'day'] = df[col_name].dt.day
    # 4. 一年中的哪一周
    df[col_name+'weekofyear'] = df[col_name].dt.weekofyear
    # 5. 一天中的哪一天
    df[col_name+'dayofyear'] = df[col_name].dt.dayofyear
    # 6. 星期几
    df[col_name+'weekday'] = df[col_name].dt.dayofweek
    # 7. 小时
    df[col_name+'hour'] = df[col_name].dt.hour
    # 8. 分钟
    df[col_name+'minute'] = df[col_name].dt.minute
    # 9. 秒
    df[col_name+'seconds'] = df[col_name].dt.second
    # df[col_name+'dayinmonth'] = df[col_name].dt.daysinmonth

    # ...

    # 1. 是否是一年的开始
    df[col_name+'is_year_start'] = df[col_name].dt.is_year_start.astype(np.int32)
    # 2. 是否是一个季度的开始
    df[col_name+'is_quarter_start'] = df[col_name].dt.is_quarter_start.astype(np.int32)
    # 3. 是否是月初
    df[col_name+'is_month_start'] = df[col_name].dt.is_month_start.astype(np.int32)
    # 4. 是否是月末
    df[col_name+'is_month_end'] = df[col_name].dt.is_month_end.astype(np.int32)
    # 5. 是否是周末
    df[col_name+'is_weekend'] = np.where(df[col_name+'weekday'].isin([5, 6]), 1, 0)


def add_len_feature(df_train, column, log1p=False):
    col = f'{column}_len'
    df_train[col] = df_train[column].apply(lambda x: len(str(x).split(',')))
    if log1p:
        df_train[col] = np.log1p(df_train[col])


def add_count_feature(train_data, column):
    new_column = f"{column}_count"
    train_data[new_column] = 0
    temp = []
    vc = train_data[column].value_counts()
    vc = (vc - vc.mean()) / vc.std()
    # vc = (vc - vc.min()) / (vc.max() - vc.min())
    for postd in train_data[column]:
        temp.append(vc[postd])
    train_data[new_column] = temp


def norm_feature(df_train, col):
    df_train[col] = (df_train[col] -
                     df_train[col].mean()) / df_train[col].std()


def add_op_feature(df_train, col1, col2, op, log1p=True, **args):
    if op == "add":
        d_train = df_train[col1] + df_train[col2]
    elif op == "sub":
        d_train = df_train[col1] - df_train[col2]
    elif op == "div":
        div_value = args.get('div_value', 1e-5)
        d_train = df_train[col1] / (df_train[col2] + div_value)
    elif op == "mul":
        d_train = df_train[col1] * df_train[col2]
    if log1p:
        d_train = np.log1p(d_train)
    df_train[col1 + op + col2] = d_train


def encode_label(df_train, df_test, col):
    le = LabelEncoder()
    le.fit(pd.concat([df_train[col], df_test[col]], sort=True).values)
    df_train[col] = le.transform(df_train[col].values)
    df_test[col] = le.transform(df_test[col].values)


def process_feature(train_data, test_data, config: Config, pseudo_data=None):
    # tf-idf vecotorize
    if config.use_tfidf:
        config.logger.info(f"Using tf-idf.")
        pass

    if config.use_w2v:
        train_vector = pd.read_csv(config.train_vector_path)
        test_vector = pd.read_csv(config.test_vector_path)
        train_data = pd.concat([train_data, train_vector], axis=1)
        test_data = pd.concat([test_data, test_vector], axis=1)
        # if config.use_pseudo:
        #     pseudo_vector = pd.read_csv(config.pseudo_vector_path)
        #     pseudo_data = pd.concat([pseudo_data, pseudo_vector], axis=1)
        #     train_data = pd.concat([train_data, pseudo_data],
        #                            ignore_index=True)
    if config.use_up_down:
        add_op_feature(train_data, 'homepage_hot_slide_up',
                       'homepage_hot_slide_down', "div")
        add_op_feature(test_data, 'homepage_hot_slide_up',
                       'homepage_hot_slide_down', "div")

    if config.use_video_play:
        add_op_feature(train_data,
                       "video_play",
                       "video_play_finish",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "video_play",
                       "video_play_finish",
                       "div",
                       div_value=1)
        add_op_feature(train_data,
                       "video_play_finish",
                       "video_play",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "video_play_finish",
                       "video_play",
                       "div",
                       div_value=1)

    if config.use_click_div_play:
        add_op_feature(train_data,
                       "click_video_play",
                       "video_play",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "click_video_play",
                       "video_play",
                       "div",
                       div_value=1)
        add_op_feature(train_data,
                       "video_play",
                       "click_video_play",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "video_play",
                       "click_video_play",
                       "div",
                       div_value=1)

    if config.play_time_div_play:
        add_op_feature(train_data,
                       "play_time",
                       "video_play",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "play_time",
                       "video_play",
                       "div",
                       div_value=1)
        add_op_feature(train_data,
                       "video_play",
                       "play_time",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "video_play",
                       "play_time",
                       "div",
                       div_value=1)

    if config.use_len_feature:
        config.logger.info(f"Using len feature")
        len_cols = ['signature']
        config.logger.info(f"len cols: {len_cols}")
        for col in len_cols:
            add_len_feature(train_data, col, log1p=True)
            add_len_feature(test_data, col, log1p=True)

    if config.use_like_dislike:
        add_op_feature(train_data, "like", "dislike", "div")
        add_op_feature(test_data, "like", "dislike", "div")
        # add_op_feature(train_data, "like", "dislike", "sub", log1p=True)
        # add_op_feature(test_data, "like", "dislike", "sub", log1p=True)

    if config.use_time_feature:
        config.logger.info(f"Using time feature.")
        train_data["create_datetime"] = pd.to_datetime(
            train_data["create_time"], unit='s')
        extract_basic_time_(train_data, "create_datetime")
        test_data["create_datetime"] = pd.to_datetime(test_data["create_time"],
                                                      unit='s')
        extract_basic_time_(test_data, "create_datetime")
        # train_data["dayofyear_"] = train_data["dayofyear"]

        train_data.drop(["create_datetime"], axis=1, inplace=True)
        test_data.drop(["create_datetime"], axis=1, inplace=True)
        # norm_feature(train_data, "create_time")
        # norm_feature(test_data, "create_time")
        # train_data.drop(["create_time"], axis=1, inplace=True)
        # test_data.drop(["create_time"], axis=1, inplace=True)

    if config.use_per_publish_feature:
        add_op_feature(train_data,
                       "fans_num_all",
                       "publish_cnt_all",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "fans_num_all",
                       "publish_cnt_all",
                       "div",
                       div_value=1)
        add_op_feature(train_data,
                       "fans_num_all",
                       "server_comment_cnt_all",
                       "div",
                       div_value=1)
        add_op_feature(test_data,
                       "fans_num_all",
                       "server_comment_cnt_all",
                       "div",
                       div_value=1)

    if config.use_log1p:
        config.logger.info(f"Using log1p.")
        log1p_columns = [
            'fans_num_all', 'publish_cnt_all', 'server_comment_cnt_all',
            'follow_num_all', 'homepage_hot_slide_down',
            'homepage_hot_slide_up', 'like', 'dislike', 'video_play', 'video_play_finish',
            'play_time', 'click_video_play',]
        if config.use_pseudo:
            log1p_columns += [
                'item_title_len_mean', 'item_title_len_max', 'item_title_len_min',
                'item_title_len_duration', 'poi_name_len_mean', 'poi_name_len_max',
                'poi_name_len_min', 'poi_name_len_duration'
            ]
        config.logger.info(f"log1p cols: {log1p_columns}")
        for col in log1p_columns:
            train_data[col] = np.log1p(train_data[col])
            test_data[col] = np.log1p(test_data[col])

    encode_label(train_data, test_data, 'gender_str')
    encode_label(train_data, test_data, 'signature')
    encode_label(train_data, test_data, 'province_1')
    encode_label(train_data, test_data, 'province_2')
    encode_label(train_data, test_data, 'province_3')

    # add_count_feature(train_data, 'user_id')
    # add_count_feature(test_data, 'user_id')

    # add_count_feature(train_data, 'product_id')
    # add_count_feature(test_data, 'product_id')

    # drop_list = ['user_id', 'post_detail']
    # train_data.drop(drop_list, axis=1, inplace=True)
    # test_data.drop(drop_list, axis=1, inplace=True)

    return train_data, test_data


def evaluate(config: Config, model, dtrain, mode="train"):
    from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
    preds = model.predict(dtrain.data, num_iteration=model.best_iteration)
    # train_predictions = [round(value) for value in train_preds]
    predictions = [int(value > config.eval_threshold) for value in preds]
    gold = dtrain.get_label()  #值为输入数据的第一行
    fbeta = fbeta_score(gold, predictions, beta=0.3)
    p = precision_score(gold, predictions)
    r = recall_score(gold, predictions)
    f1 = f1_score(gold, predictions)
    if config.use_log:
        config.logger.info(
            f"{mode} {config.eval_threshold} p: {p:.4f} r: {r:.4f} f1: {f1:.4f} fbeta: {fbeta:.4f}"
        )
    return fbeta, p, r, f1


def evaluate_tabnnet(config: Config, model, dtrain, mode="train"):
    from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
    predictions = model.predict(dtrain[0])

    # train_predictions = [round(value) for value in train_preds]
    # predictions = [int(value > config.eval_threshold) for value in preds]
    gold = dtrain[1]  #值为输入数据的第一行
    fbeta = fbeta_score(gold, predictions, beta=0.3)
    p = precision_score(gold, predictions)
    r = recall_score(gold, predictions)
    f1 = f1_score(gold, predictions)
    config.logger.info(
        f"{mode} {config.eval_threshold} p: {p:.4f} r: {r:.4f} f1: {f1:.4f} fbeta: {fbeta:.4f}"
    )
    return fbeta, p, r, f1


def get_train_and_dev_data(config: Config, mode="split"):
    if mode == "test":
        total_data = pd.read_csv(config.test_path)
    else:
        total_data = pd.read_csv(config.train_path)

    # drop duplicate
    # total_data = pd.concat([total_data, total_vector], axis=1)
    # total_data.drop_duplicates(subset=["request_id", "user_id", 'post_detail'], keep='last', inplace=True)

    # drop_list = ['label']
    drop_list = config.drop_list
    total_x = total_data.drop(config.drop_list, axis=1)

    total_y = total_data['label']
    config.logger.info(f"Total: {total_x.shape} {total_y.shape}")
    if mode == "split":
        from sklearn.model_selection import train_test_split
        train_x, dev_x, train_y, dev_y = train_test_split(
            total_x, total_y, test_size=0.1, random_state=config.random_seed)
        config.logger.info(
            f"Train: {train_x.shape} {train_y.shape} Dev: {dev_x.shape} {dev_y.shape}"
        )
        return train_x, train_y, dev_x, dev_y
    return total_x, total_y


def get_train_and_test_data(config: Config):
    train_data = pd.read_csv(config.train_path)
    test_data = pd.read_csv(config.test_path)
    if config.use_pseudo:
        pseudo_data = pd.read_csv(config.pseudo_path)
        train_data = pd.concat([train_data, pseudo_data])
        train_data.index = train_data.reset_index()
    # if config.use_pseudo:
    #     pseudo_data = pd.read_csv(config.pseudo_path)
    #     return train_data, test_data, pseudo_data

    return train_data, test_data


def get_pseudo_data(config: Config):
    pseudo_data = pd.read_csv(config.pseudo_path)
    pseudo_x = pseudo_data.drop(config.drop_list, axis=1)  # 训练集输入
    pseudo_y = pseudo_data['label']  # 训练集标签
    return pseudo_x, pseudo_y


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def save_model(config: Config, model, path=None):

    if config.model_type == "xgb":
        save_path = config.xgb_save_path
        model.save_model()
    elif config.model_type == "lgb":
        save_path = config.lgb_save_path
    elif config.model_type == "tab":
        save_path = config.tab_save_path
    if path is not None:
        save_path = path
    model.save_model(save_path)
    config.logger.info(f"Saved model at: {save_path}")


def load_model(config: Config = None, path=None):
    save_path = config.lgb_save_path if path is None else path
    bst = lgb.Booster(model_file=save_path)
    if config is not None:
        config.logger.info(f"Load model from: {save_path}")
    return bst