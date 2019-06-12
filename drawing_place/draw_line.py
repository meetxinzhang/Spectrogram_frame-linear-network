from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 加载日志数据
ea_09 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over09/train/events.out.tfevents.1560330514.localhost.localdomain')
ea_09.Reload()
ea_08 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/2019-04-24-15-20-54/train/events.out.tfevents.1556090456.localhost.localdomain')
ea_08.Reload()
ea_07 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over07/train/events.out.tfevents.1556630677.DEVIN-ENTERPRIS')
ea_07.Reload()
ea_06 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over06/train/events.out.tfevents.1556629044.DEVIN-ENTERPRIS')
ea_06.Reload()
ea_05 = event_accumulator.EventAccumulator('D:/GitHub/ProjectX/tensor_logs/over05/train/events.out.tfevents.1556628694.DEVIN-ENTERPRIS')
ea_05.Reload()
print(ea_08.scalars.Keys())

line_name = 'loss'
line_09 = ea_09.scalars.Items(line_name)
line_08 = ea_08.scalars.Items(line_name)
line_07 = ea_07.scalars.Items(line_name)
line_06 = ea_06.scalars.Items(line_name)
line_05 = ea_05.scalars.Items(line_name)
print(len(line_08))

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)


# ax1.set_xlim(0)

ax1.plot([i.step for i in line_09], [i.value for i in line_09], label='90%')
ax1.plot([i.step for i in line_08], [i.value for i in line_08], label='80%')
ax1.plot([i.step for i in line_07], [i.value for i in line_07], label='70%')
ax1.plot([i.step for i in line_06], [i.value for i in line_06], label='60%')
ax1.plot([i.step for i in line_05], [i.value for i in line_05], label='50%')

# 设置刻度字体大小
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax1.set_xlabel("step", fontsize=18)
ax1.set_ylabel(line_name, fontsize=18)

plt.legend(loc='lower right', fontsize=18)
plt.show()
