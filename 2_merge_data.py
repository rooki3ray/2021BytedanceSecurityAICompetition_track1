import pandas as pd
import numpy as np
from utils import extract_basic_time_

def merge_user_and_basic(user_id_path, basic_info_path, dst_path):

    df_basic = pd.read_csv(basic_info_path)
    df_user = pd.read_csv(user_id_path)
    if 'label' not in df_user.columns:
        df_user['label'] = -1

    print(df_basic.shape, df_user.shape)

    df_dst = pd.merge(df_basic, df_user, on=['id'], how='outer')

    print(df_dst.shape)

    df_dst.to_csv(dst_path, index=False)


def merge_bahavior(src_path, behavior_path, dst_path):
    df_src = pd.read_csv(src_path)
    df_behavior = pd.read_csv(behavior_path)
    # if 'label' not in df_behavior.columns:
    #     df_behavior['label'] = -1

    print(df_src.shape, df_behavior.shape)

    df_dst = pd.merge(df_src, df_behavior, on=['id'], how='outer')

    print(df_dst.shape)

    df_dst.to_csv(dst_path, index=False)

def gen_statistic(df_grpid, df_dst, column_name):
    dst_mean = df_grpid[[column_name]].agg("mean")
    dst_max = df_grpid[[column_name]].agg("max")
    dst_min = df_grpid[[column_name]].agg("min")
    dst_duration = dst_max - dst_min

    df_dst[column_name+"_mean"] = dst_mean
    df_dst[column_name+"_max"] = dst_max
    df_dst[column_name+"_min"] = dst_min
    df_dst[column_name+"_duration"] = dst_duration


def merge_submit_notext(src_path, submit_path, dst_path, mode="train", TRAIN_ITEM_CREATIME_MEAN=None):
    def unique(provinces):
        provinces = list(provinces)
        prov_unq, prov_cnt = np.unique(provinces, return_counts=True)
        idx = np.argsort(-prov_cnt)
        return prov_unq[idx].tolist()

    df_src = pd.read_csv(src_path)
    df_submit = pd.read_csv(submit_path)

    # item_create_time = df_submit["item_create_time"]
    # df_submit[["item_create_time"]] = (item_create_time - item_create_time.mean()) / item_create_time.std()
    df_submit["item_create_time_"] = pd.to_datetime(
            df_submit["item_create_time"], unit='s')

    extract_basic_time_(df_submit, "item_create_time_")
    df_submit.drop('item_create_time_', axis=1, inplace=True)
    # if mode == "train":
    #     TRAIN_ITEM_CREATIME_MEAN = [
    #                 df_submit["item_create_time_dayofyear"].mean(),
    #                 df_submit["item_create_time_dayofyear"].mode().to_numpy(),
    #                 # df_submit["item_create_time_weekofyear"].mode().to_numpy()
    #         ]
    # else:
    #     # df_submit["item_create_time_dayofyear"] = df_submit["item_create_time_dayofyear"] - \
    #     #                             df_submit["item_create_time_dayofyear"].mode().to_numpy() + \
    #     #                             TRAIN_ITEM_CREATIME_MEAN[0]
    #     # df_submit["item_create_time_dayofyear"] = df_submit["item_create_time_dayofyear"] - \
    #     #                             df_submit["item_create_time_dayofyear"].mean() + \
    #     #                             TRAIN_ITEM_CREATIME_MEAN[0]
    #     # df_submit["item_create_time_weekofyear"] = (df_submit["item_create_time_weekofyear"] - \
    #     #                             df_submit["item_create_time_weekofyear"].mode().to_numpy() + \
    #     #                             TRAIN_ITEM_CREATIME_MEAN[1])

    df_submit["item_create_time_dayofyear"] = (df_submit["item_create_time_dayofyear"] - \
        df_submit["item_create_time_dayofyear"].min()) / \
            (df_submit["item_create_time_dayofyear"].max() - \
                df_submit["item_create_time_dayofyear"].min())
    df_submit["poi_name_len"] = df_submit["poi_name"].apply(lambda x:
                                        0 if "nan" in str(x) else len(str(x).split(",")))
    df_submit["item_title_len"] = df_submit["item_title"].apply(lambda x:
                                        0 if "nan" in str(x) else len(str(x).split(",")))
    # df_submit["item_create_time_dayofyear"] = (df_submit["item_create_time_dayofyear"] - \
    #     df_submit["item_create_time_dayofyear"].mean()) / \
    #         df_submit["item_create_time_dayofyear"].std()

    df_submit_grpid = df_submit.groupby("id")

    # province
    train_province = df_submit_grpid[["item_province_cn"]].agg(unique)
    # train_province["province_nums"] = [len(i) \
    #     for i in train_province["item_province_cn"]]
    train_province["province_1"] = [i[0] \
        for i in train_province["item_province_cn"]]
    train_province["province_2"] = [i[1] if len(i) >= 2 else "NULL" \
        for i in train_province["item_province_cn"]]
    train_province["province_3"] = [i[2] if len(i) >= 3 else "NULL" \
        for i in train_province["item_province_cn"]]
    if 'item_province_cn' in train_province:
        train_province.drop('item_province_cn', axis=1, inplace=True)

    # times
    submit_times = df_submit_grpid[["id"]].agg(len)
    train_province["submit_times"] = submit_times["id"]
    # train_province["item_create_time_weekday"] = df_submit["item_create_time_weekday"]

    # create_time
    gen_statistic(df_submit_grpid, train_province, "item_create_time")
    gen_statistic(df_submit_grpid, train_province, "item_title_len")
    gen_statistic(df_submit_grpid, train_province, "poi_name_len")

    # gen_statistic(df_submit_grpid, train_province, "item_create_time_weekday")
    gen_statistic(df_submit_grpid, train_province, "item_create_time_dayofyear")
    # gen_statistic(df_submit_grpid, train_province, "item_create_time_weekofyear")

    # gen_statistic(df_submit_grpid, train_province, "item_create_time_day")
    # gen_statistic(df_submit_grpid, train_province, "item_create_time_hour")

    # merge
    print(df_src.shape, train_province.shape)

    df_dst = pd.merge(df_src, train_province, on=['id'], how='outer')
    # extract_basic_time_(df_submit, "item_create_time_")

    # df_dst["item_create_time_mean_"] = pd.to_datetime(
    #         df_dst["item_create_time_mean"], unit='s')
    # extract_basic_time_(df_dst, "item_create_time_mean_")
    # df_dst.drop('item_create_time_mean_', axis=1, inplace=True)


    df_dst["item_create_time_mean"] = df_dst["item_create_time_mean"] - df_src["create_time"]
    df_dst["item_create_time_max"] = df_dst["item_create_time_max"] - df_src["create_time"]
    df_dst["item_create_time_min"] = df_dst["item_create_time_min"] - df_src["create_time"]
    print(df_dst.shape)

    df_dst.to_csv(dst_path, index=False)
    return TRAIN_ITEM_CREATIME_MEAN

def merge_sig_vec(src_path, sig_vec_path, dst_path):
    df_src = pd.read_csv(src_path)
    df_behavior = pd.read_csv(sig_vec_path)
    # if 'label' not in df_behavior.columns:
    #     df_behavior['label'] = -1

    print(df_src.shape, df_behavior.shape)

    df_dst = pd.merge(df_src, df_behavior, on=['id'], how='left')

    print(df_dst.shape)

    df_dst.to_csv(dst_path, index=False)

if __name__ == '__main__':
    dst_train = './data/train.csv'
    dst_test = './data/test.csv'
    merge_user_and_basic("./data/raw/训练数据/用户标签.csv",
                         "./data/raw/训练数据/用户基础信息.csv", dst_train)
    merge_user_and_basic("./data/raw/测试数据/用户id(无标签).csv",
                         "./data/raw/测试数据/用户基础信息.csv", dst_test)
    merge_bahavior(dst_train, './data/raw/训练数据/用户行为信息.csv', dst_train)
    merge_bahavior(dst_test, './data/raw/测试数据/用户行为信息.csv', dst_test)
    mean = merge_submit_notext(dst_train, './data/raw/训练数据/用户投稿信息.csv', dst_train)
    merge_submit_notext(dst_test, './data/raw/测试数据/用户投稿信息.csv', dst_test, "test", mean)
    merge_sig_vec(dst_train, './data/train_signature_vec.csv', dst_train)
    merge_sig_vec(dst_test, "./data/test_signature_vec.csv", dst_test)

    # merge_sig_vec(dst_train, './data/train_item_title_vec_mean.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_item_title_vec_mean.csv", dst_test)
    # merge_sig_vec(dst_train, './data/train_item_title_vec_max.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_item_title_vec_max.csv", dst_test)
    # merge_sig_vec(dst_train, './data/train_item_title_vec_min.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_item_title_vec_min.csv", dst_test)

    # merge_sig_vec(dst_train, './data/train_poi_name_vec_mean.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_poi_name_vec_mean.csv", dst_test)
    # merge_sig_vec(dst_train, './data/train_poi_name_vec_max.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_poi_name_vec_max.csv", dst_test)
    # merge_sig_vec(dst_train, './data/train_poi_name_vec_min.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_poi_name_vec_min.csv", dst_test)
    # merge_sig_vec(dst_train, './data/train_sig_fasttext.csv', dst_train)
    # merge_sig_vec(dst_test, "./data/test_sig_fasttext.csv", dst_test)
