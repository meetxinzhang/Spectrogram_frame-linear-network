"""
混淆矩阵分析
"""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# matplotlib.rcParams['font.size'] = 18


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

    print(data)
    file.close()
    return data


data = txt_read()
y_pred = data[0]
y_true = data[1]
target_names = ['Anser anser', 'Buteo buteo', 'Oriolus oriolus', 'Pica pica']
labels = ['0', '1', '2', '3']
print(classification_report(y_pred=y_pred, y_true=y_true, target_names=target_names))

cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)
plt.matshow(cm)
# 画混淆矩阵图，配色风格使用
plt.colorbar()
# 颜色标签
for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        # 坐标轴标签
        plt.xlabel('Predicted label')
        # 坐标轴标签

plt.show()
