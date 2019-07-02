from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# # 加载日志数据
# ea_09 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over09_f/train/events.out.tfevents.1556092927.localhost.localdomain')
# ea_09.Reload()
# ea_08 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/2019-04-24-15-20-54/train/events.out.tfevents.1556090456.localhost.localdomain')
# ea_08.Reload()
# ea_07 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over07/train/events.out.tfevents.1556630677.DEVIN-ENTERPRIS')
# ea_07.Reload()
# ea_06 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over06/train/events.out.tfevents.1556629044.DEVIN-ENTERPRIS')
# ea_06.Reload()
# ea_05 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over05/train/events.out.tfevents.1556628694.DEVIN-ENTERPRIS')
# ea_05.Reload()
# print(ea_08.scalars.Keys())
#
# line_name = 'accuracy'
# line_09 = ea_09.scalars.Items(line_name)
# line_08 = ea_08.scalars.Items(line_name)
# line_07 = ea_07.scalars.Items(line_name)
# line_06 = ea_06.scalars.Items(line_name)
# line_05 = ea_05.scalars.Items(line_name)
# print(len(line_08))


def txt_read(logs_path):
    loss_history = []
    acc_history = []
    test_loss_history = []
    test_acc_history = []
    data_m = [loss_history, acc_history, test_loss_history, test_acc_history]

    file = open(logs_path, 'r')
    i_line = 0
    for line in file.readlines():
        line = line.strip('\n')
        for v in line.split('\t'):
            if v != '':
                data_m[i_line].append(v)
        i_line += 1

    file.close()
    return data_m


line_name = 'accuracy'
data_0 = txt_read('tensor_logs/lines2019-06-27-18-15-39.txt')
data_25 = txt_read('tensor_logs/lines_over25.txt')
data_5 = txt_read('tensor_logs/lines_over5.txt')
len_test = len(data_0[3])

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)


# ax1.set_xlim(0)

ax1.plot([float(i) for i in range(len_test)], [float(i) for i in data_0[3]], label='0.0')
ax1.plot([float(i) for i in range(len_test)], [float(i) for i in data_25[3]], label='0.25')
ax1.plot([float(i) for i in range(len_test)], [float(i) for i in data_5[3]], label='0.50')


# 设置刻度字体大小
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax1.set_xlabel("step", fontsize=18)
ax1.set_ylabel(line_name, fontsize=18)

plt.legend(loc='lower right', fontsize=18)
plt.show()
