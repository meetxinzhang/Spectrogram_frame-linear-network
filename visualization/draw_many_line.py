# coding: utf-8
# ---
# @File: draw_many_line.py
# @description: 用 matplotlib 在一幅图上绘多条折线，从读取 tensorboard 保存的数据，其在 eager_main 中保存
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 1月11, 2019
# ---

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'Times New Roman'


# 加载tensorboard日志数据
ea_09 = event_accumulator.EventAccumulator('tensor_logs/over09_f/test/events.out.tfevents.1556093081.localhost.localdomain')
ea_09.Reload()
ea_08 = event_accumulator.EventAccumulator('tensor_logs/over08/test/events.out.tfevents.1556090611.localhost.localdomain')
ea_08.Reload()
ea_07 = event_accumulator.EventAccumulator('tensor_logs/over07/test/events.out.tfevents.1556630931.DEVIN-ENTERPRIS')
ea_07.Reload()
ea_06 = event_accumulator.EventAccumulator('tensor_logs/over06/test/events.out.tfevents.1556629252.DEVIN-ENTERPRIS')
ea_06.Reload()
ea_05 = event_accumulator.EventAccumulator('tensor_logs/over05/test/events.out.tfevents.1556628878.DEVIN-ENTERPRIS')
ea_05.Reload()
print(ea_08.scalars.Keys())

line_name = 'Loss'
index_line = 'loss'
line_09 = ea_09.scalars.Items(index_line)
line_08 = ea_08.scalars.Items(index_line)
line_07 = ea_07.scalars.Items(index_line)
line_06 = ea_06.scalars.Items(index_line)
line_05 = ea_05.scalars.Items(index_line)
print(len(line_08))


# def txt_read(logs_path):
#     loss_history = []
#     acc_history = []
#     test_loss_history = []
#     test_acc_history = []
#     data_m = [loss_history, acc_history, test_loss_history, test_acc_history]
#
#     file = open(logs_path, 'r')
#     i_line = 0
#     for line in file.readlines():
#         line = line.strip('\n')
#         for v in line.split('\t'):
#             if v != '':
#                 data_m[i_line].append(v)
#         i_line += 1
#
#     file.close()
#     return data_m
#
#
# line_name = 'Accuracy'
# index_line = 3
# # over in 6
# data_0 = txt_read('tensor_logs/lines2019-06-27-18-15-39.txt')
# data_25 = txt_read('tensor_logs/lines_over25.txt')
# data_5 = txt_read('tensor_logs/lines2019-06-27-20-21-23.txt')
# len_test = len(data_0[index_line])

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)


# # over in txt
# ax1.plot([float(i) for i in range(len_test)], [float(i) for i in data_0[index_line]], label='0.0')
# ax1.plot([float(i) for i in range(len_test)], [float(i) for i in data_25[index_line]], label='0.25')
# ax1.plot([float(i) for i in range(len_test)], [float(i) for i in data_5[index_line]], label='0.50')
# over in tensorboard
ax1.plot([i.step for i in line_09], [i.value for i in line_09], label='0.9')
ax1.plot([i.step for i in line_08], [i.value for i in line_08], label='0.8')
ax1.plot([i.step for i in line_07], [i.value for i in line_07], label='0.7')
ax1.plot([i.step for i in line_06], [i.value for i in line_06], label='0.6')
ax1.plot([i.step for i in line_05], [i.value for i in line_05], label='0.5')

# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
ax1.set_xlabel("Step")
ax1.set_ylabel(line_name)

plt.legend(loc='lower right')
plt.show()
