import time
from sklearn.svm import SVC
import numpy as np
import pickle

def train(bag_of_words_path="data/generate/train_bag_of_words.npy",
          label_path="data/generate/train_label.npy", save_model_path="data/svm.pickle"):
    """
    使用svm模型训练的方法
    :param bag_of_words_path:训练集词袋模型的路径
    :param label_path:训练集标签的路径
    :param save_model_path:模型保存的路径
    """
    print('开始训练~')
    # 先读取词袋模型数据与标签数据
    train_data = np.load(bag_of_words_path).astype(np.float32)
    train_label = np.load(label_path).astype(np.int32)
    print('读取数据完成~')


    # 初始化并设置svm分类器
    svm = SVC(kernel='rbf', C=1000, decision_function_shape='ovo')

    # 开始训练
    print('开始训练svm模型···')
    start = time.time()
    svm.fit(train_data, train_label)
    end = time.time()
    print(f'训练完成，共耗时{end - start}s，模型结果与划分的训练集、测试集已保存到磁盘中。')

    with open(save_model_path, 'wb') as f:
        pickle.dump(svm, f)

if __name__ == '__main__':
    train()
