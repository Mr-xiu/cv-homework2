import cv2
import numpy as np
import time
import os
from sklearn.cluster import MiniBatchKMeans


class Generate:
    def __init__(self):
        # self.dataset = []  # 数据
        # self.label = []  # 特征
        self.test_data = []
        self.train_data = []
        self.test_label = []
        self.train_label = []
        self.train_bag_of_words = []  # 词袋模型特征向量
        self.test_bag_of_words = []  # 词袋模型特征向量

    def run(self, data_path='15-Scene', clusters_num=500):
        print('开始生成···')
        start = time.time()
        self.load_img_and_sift(data_path)
        end = time.time()
        print(f'读取图片并提取SIFT特征步骤完成，共耗时{end - start}s~')

        start = time.time()
        kmeans = self.kmeans(clusters_num)
        end = time.time()
        print(f'kmeans聚类步骤完成，共耗时{end - start}s~')

        start = time.time()
        self.generate_bag_of_words(kmeans)
        end = time.time()
        print(f'构建词袋向量步骤完成，共耗时{end - start}s~')

        # 保存结果
        self.save_result()
        print('生成的结果已保存~')

    def load_img_and_sift(self, data_path):
        """
        读取数据集文件并提取图片的SIFT特征
        :param data_path: 数据集的根路径
        """
        sift = cv2.SIFT_create()
        sub_folder_list = os.listdir(data_path)
        for sub_folder in sub_folder_list:
            file_list = os.listdir(data_path + '/' + sub_folder)
            i = 1
            for file in file_list:
                # 读取图片
                img = cv2.imread(data_path + '/' + sub_folder + '/' + file)
                # 转换为灰度图
                img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(img_grey, None)
                if i <= 150:
                    self.train_data.append(des)
                    self.train_label.append(sub_folder)
                else:
                    self.test_data.append(des)
                    self.test_label.append(sub_folder)
                i += 1

    def kmeans(self, clusters_num):
        """
        对sift生成的图片特征进行kmeans聚类的方法
        :param clusters_num: 聚类的簇数
        :return 产生的聚类模型
        """
        kmeans_data = np.vstack(self.train_data)  # 将不同图像的特征合起来
        kmeans = MiniBatchKMeans(n_clusters=clusters_num)
        kmeans.fit(kmeans_data)
        return kmeans

    def generate_bag_of_words(self, kmeans):
        """
        创建词袋模型，将SIFT特征转换为词袋向量
        :param kmeans: 聚类模型
        """
        for data in self.train_data:
            class_list = kmeans.predict(data)
            # 生成词袋模型
            hist, _ = np.histogram(class_list, bins=np.arange(kmeans.n_clusters + 1))
            self.train_bag_of_words.append(hist)
        for data in self.test_data:
            class_list = kmeans.predict(data)
            # 生成词袋模型
            hist, _ = np.histogram(class_list, bins=np.arange(kmeans.n_clusters + 1))
            self.test_bag_of_words.append(hist)

    def save_result(self):
        np.save('data/generate/train_bag_of_words.npy', np.array(self.train_bag_of_words))
        np.save('data/generate/train_label.npy', np.array(self.train_label))
        np.save('data/generate/test_bag_of_words.npy', np.array(self.test_bag_of_words))
        np.save('data/generate/test_label.npy', np.array(self.test_label))


if __name__ == '__main__':
    generate = Generate()
    generate.run(data_path='15-Scene', clusters_num=500)
