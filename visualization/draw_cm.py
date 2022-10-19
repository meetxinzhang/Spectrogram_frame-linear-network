# coding: utf-8
# ---
# @File: draw_cm.py
# @description: 混淆矩阵分析, 从txt文件中获得实验日志数据，其在 eager_main 中保存
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 6月27, 2019
# ---

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'Times New Roman'


def txt_read():
    logs_path = 'tensor_logs/y_2019-06-27-20-21-23.txt'

    y_pred = []
    y_true = []
    data = [y_pred, y_true]

    file = open(logs_path, 'r')
    i_line = 0
    for line in file.readlines():
        line = line.strip('\n')
        for v in line.split('\t'):
            if v != '':
                data[i_line].append(v)
        i_line += 1

    file.close()
    return data


def plot_matrix(cm, classes, title=None, cmap=plt.cm.Blues):

    # 按行进行归一化
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    # for row in str_cm:
    #     print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if int(cm[i, j] * 100 + 0.5) == 0:
    #             cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]) >= 0:
                ax.text(j, i, format(int(cm[i, j]), 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.savefig('cm.jpg', dpi=300)
    plt.show()


data = txt_read()
y_pred = data[0]
y_true = data[1]
target_names = ['Anser anser', 'Buteo buteo', 'Oriolus oriolus', 'Pica pica']
labels = ['0', '1', '2', '3']
print(classification_report(y_pred=y_pred, y_true=y_true, target_names=target_names, digits=3))

cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)
print(cm)

# plt.matshow(cm)
# # 画混淆矩阵图，配色风格使用
# plt.colorbar()
# # 颜色标签
# for x in range(len(cm)):
#     for y in range(len(cm)):
#         plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
#         plt.ylabel('True label')
#         # 坐标轴标签
#         plt.xlabel('Predicted label')
#         # 坐标轴标签
#
# plt.show()


plot_matrix(cm, target_names)
