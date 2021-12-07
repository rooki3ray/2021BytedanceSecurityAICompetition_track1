import logging
import multiprocessing
import os.path
import sys
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from tqdm import tqdm
import pandas as pd 
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
# logging.getLogger().setLevel(logging.INFO)

def train_word2vec(input_dir, object_name, seed=1234, vector_size=50, window=10, min_count=1, epochs=50):
    word2vec = f'data/word2vec_{object_name}'
    word2vec_format = f'data/word2vec_format_{object_name}'
    # 训练模型 
    model = Word2Vec(PathLineSentences(input_dir),
                        vector_size=vector_size, window=window, min_count=min_count, 
                        sample=0.002, sg=1, negative=2,
                        workers=multiprocessing.cpu_count(), epochs=epochs, seed=seed)
    print("Train Success! Saving Model...")
    model.save(word2vec)
    model.wv.save_word2vec_format(word2vec_format, binary=False)
    return model


def gen_vec(
    object_name = "signature",
    seed=1234):
    train_path_dict = {
        "signature": "./data/raw/训练数据/用户基础信息.csv",
        "poi_name": "./data/raw/训练数据/用户投稿信息.csv",
        "item_title": "./data/raw/训练数据/用户投稿信息.csv",
    }
    test_path_dict = {
        "signature": "./data/raw/测试数据/用户基础信息.csv",
        "poi_name": "./data/raw/测试数据/用户投稿信息.csv",
        "item_title": "./data/raw/测试数据/用户投稿信息.csv",
    }
    train_path = train_path_dict[object_name]
    test_path = test_path_dict[object_name]
    if not os.path.exists("data/sentence"):
        os.mkdir("data/sentence")
    sentence_path = os.path.join("data/sentence", object_name)
    if not os.path.exists(sentence_path):
        os.mkdir(sentence_path)
    all_sentence_path = f"{sentence_path}/{object_name}_word2vec"
    train_sentence_path = f"{sentence_path}/train_{object_name}_word2vec"
    test_sentence_path = f"{sentence_path}/test_{object_name}_word2vec"

    # read data
    logging.info("Read data from csv...")
    train_data = pd.read_csv(train_path)

    test_data = pd.read_csv(test_path)

    train_x = train_data[["id",object_name]]
    # train_y = test_data[['label']]

    test_x = test_data[["id",object_name]]
    # test_y = test_data['label']
    print(f"Train:{train_x.shape[0]}  Test:{test_x.shape[0]}")

    # concat train and test
    print("Concat train and test for w2v...")
    test_x.index = test_x.index + train_x.shape[0]

    data = pd.concat([train_x, test_x], axis=0)

    data[object_name] = data[object_name].map(str).apply(lambda x: x.replace(",", " "))
    
    data[[object_name]].to_csv(all_sentence_path, index=False, header=False)
    data[:train_x.shape[0]].to_csv(train_sentence_path, index=False, header=False)
    data[train_x.shape[0]:].to_csv(test_sentence_path, index=False, header=False)
    print("Train word2vec...")
    if os.path.exists(f"./data/word2vec_{object_name}"):
        model = gensim.models.word2vec.Word2Vec.load(f"./data/word2vec_{object_name}")
    else:
        model = train_word2vec(all_sentence_path, object_name, 1234)

    print("Calculate sentence mean vectors...")
    vectors_train = []
    id_train = []
    vectors_test = []
    id_test = []

    for i in tqdm(train_x.index):
        sentence = data[object_name][i]
        id_train.append(data['id'][i])
        words = sentence.split(" ")
        vectors_train.append(np.mean([model.wv.get_vector(word) for word in words], axis=0))
    for i in tqdm(test_x.index):
        sentence = data[object_name][i]
        id_test.append(data['id'][i])
        words = sentence.split(" ")
        vectors_test.append(np.mean([model.wv.get_vector(word) for word in words], axis=0))
    vectors_train = np.array(vectors_train)
    vectors_test = np.array(vectors_test)
    print("Saving sentence vectors...")

    vectors_train_dict = {f"vector_{object_name}_{i}" : vectors_train[:, i] for i in range(vectors_train.shape[1])}
    vectors_test_dict = {f"vector_{object_name}_{i}" : vectors_test[:, i] for i in range(vectors_test.shape[1])}
    vectors_train_dict["id"] = id_train
    vectors_test_dict["id"] = id_test

    vectors_train_pd = pd.DataFrame(vectors_train_dict)
    vectors_test_pd = pd.DataFrame(vectors_test_dict)

    vectors_train_pd.to_csv(f"data/train_{object_name}_vec.csv", index=False)
    vectors_test_pd.to_csv(f"data/test_{object_name}_vec.csv", index=False)
    # vectors_train.tofile("data/train_vectors.npy")
    # vectors_test.tofile("data/test_vectors.npy")
    print(f"Vectors saved in data/train_{object_name}_vec.csv and data/test_{object_name}_vec.csv.")

if __name__ == '__main__':
    gen_vec("signature", 1234)