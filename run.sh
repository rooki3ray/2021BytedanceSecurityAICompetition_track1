set -x

python 1_word2vec.py
python 2_merge_data.py
python 3_5_train_kfold.py
python 4_pseudo_label.py
python 3_5_train_kfold.py --pseudo
