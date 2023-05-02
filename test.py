import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix
def test(data_path='15-Scene',model_path="data/svm.pickle", test_data_path="data/train/test_data.npy", test_label_path="data/train/test_label.npy", conf_matrix_path="data/cnf_matrix.png"):
    """
    根据训练得到的svm模型在测试集上测试并生成混淆矩阵的方法
    :param data_path:数据集的根目录
    :param model_path:训练生成的SVM模型的路径
    :param test_data_path:测试数据的路径
    :param test_label_path:测试标签的路径
    :param conf_matrix_path:混淆矩阵储存的路径
    """
    print('开始测试~')
    # 先读取测试集数据
    test_data = np.load(test_data_path).astype(np.float32)
    test_label = np.load(test_label_path).astype(np.int32)
    
    # 读取svm模型
    with open(model_path, 'rb') as f:
        svm = pickle.load(f)

    print(test_data.shape)
    print(test_label.shape)
    # 进行预测
    pred_label = svm.predict(test_data)

    print('各类别的测试结果如下：')
    # 输出各类别的测试结果
    print(classification_report(test_label, pred_label))

    # 生成混淆矩阵并将其保存在磁盘中
    conf_matrix = confusion_matrix(test_label, pred_label)
    cnf_matrix_norm = conf_matrix.astype('float') / \
                      conf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix_norm = np.around(cnf_matrix_norm, decimals=2)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cnf_matrix_norm, annot=True, cmap='Blues')
    class_list = os.listdir(data_path)
    plt.ylim(0, 15)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=90)
    plt.yticks(tick_marks, class_list, rotation=45)
    plt.savefig(conf_matrix_path)
    plt.show()
    print(f'测试生成的混淆矩阵已保存到了{conf_matrix_path}中~')


if __name__ == '__main__':
    test()
