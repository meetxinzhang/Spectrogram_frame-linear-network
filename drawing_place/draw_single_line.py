from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 18

# 加载日志数据
ea = event_accumulator.\
    EventAccumulator('D:/GitHub/ProjectX/tensor_logs/2019-06-14-11-39-58/train/events.out.tfevents.1560483598.localhost.localdomain.v2')
ea.Reload()

print(ea.scalars.Keys())

line_name = 'loss'
line = ea.scalars.Items(line_name)

print(len(line))

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)

ax1.plot([i.step for i in line], [i.value for i in line], label='test')

# 设置刻度字体大小
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax1.set_xlabel("step", fontsize=18)
ax1.set_ylabel(line_name, fontsize=18)

plt.legend(loc='lower right', fontsize=18)
plt.show()
