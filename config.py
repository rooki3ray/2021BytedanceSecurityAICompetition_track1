import logging
import os
import time


class Config:
    def __init__(self, use_log=True, use_pseudo=False, gen_pseudo=False) -> None:
        super(Config, self).__init__()
        self.random_seed = 1234
        self.model_type = "lgb"

        self.train_path = "./data/train.csv"
        self.test_path = "./data/test.csv"
        self.pseudo_path = "./data/pseudo.csv"

        self.use_log = use_log
        self.drop_list = ['id', 'label', 'feed_request']
        self.threshold = 0.95
        self.eval_threshold = 0.95

        self.test_threshold = 0.999 if gen_pseudo else 0.985  # .8

        cur_time = time.strftime("%m%d_%H%M", time.localtime())

        self.use_w2v = False
        self.use_tfidf = False
        self.use_lda = False
        self.use_pseudo = use_pseudo
        self.use_len_feature = True
        self.use_vote = False
        self.use_log1p = True
        self.use_time_feature = True
        self.num_fold = 10
        self.use_per_publish_feature = True
        self.use_like_dislike = True
        self.use_video_play = True
        self.use_click_div_play = True
        self.play_time_div_play = True
        self.use_up_down = True
        # LightGBM

        self.lgb_params = {
            'seed': 1745,
            'learning_rate': 0.1,
            'lambda_l1': 0.0,
            'lambda_l2': 1,
            'max_depth': 14,  # 8
            "max_bin": 20,
            "n_jobs": 4,
            "is_unbalance": True,
            'min_child_weight': 0.01,
            'num_leaves': 160,  # 32
            'bagging_fraction': 0.8,
            "feature_fraction": 0.6,  # 0.8
            'bagging_freq': 2,
            'verbose': -1,
            'objective': 'binary',  # 目标函数
            'metric': ['auc', 'binary_logloss']
        }
        self.lgb_num_round = 10000
        self.lgb_save_path = "./saved/lgb.bin"
        if self.use_pseudo:
            self.cur_dir = os.path.join(
                "saved", f"{cur_time}_pseudo_{self.test_threshold}")
        else:
            self.cur_dir = os.path.join("saved",
                                        f"{cur_time}_{self.test_threshold}")
        # self.result_path = f"./saved/{cur_time}_{self.test_threshold}_results.csv"
        self.result_path = os.path.join(
            self.cur_dir, f"{cur_time}_{self.test_threshold}_results.csv")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists("saved"):
            os.mkdir("saved")
        if self.use_log:
            if not os.path.exists(self.cur_dir):
                os.mkdir(self.cur_dir)
            os.system(f"cp config.py {self.cur_dir}/")
            os.system(f"cp merge_data.py {self.cur_dir}/")
            os.system(f"cp utils.py {self.cur_dir}/")
            os.system(f"cp evaluate_kfold.py {self.cur_dir}/")
            os.system(f"cp train_kfold.py {self.cur_dir}/")
            self.init_logger()
            self.print_params()

    def print_params(self):
        self.logger.info("-" * 50)
        for k, v in vars(self).items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("-" * 50)

    def init_logger(self):
        handler = logging.FileHandler(os.path.join(self.cur_dir, 'log.log'))
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(fmt=formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(sh)