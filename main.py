from generate import Generate
from train import train
from test import test
if __name__ == '__main__':
    # 生成
    generate = Generate()
    generate.run(data_path='15-Scene', clusters_num=500)

    # 训练
    train()

    # 测试
    test()