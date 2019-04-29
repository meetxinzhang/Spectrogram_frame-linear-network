from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 加载日志数据
ea = event_accumulator.EventAccumulator('tensor_logs/id=2/test/events.out.tfevents.1556029340.localhost.localdomain')
ea.Reload()
print(ea.scalars.Keys())

val_acc = ea.scalars.Items('val_acc')
print(len(val_acc))
print([(i.step, i.value) for i in val_acc])

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
val_acc = ea.scalars.Items('val_acc')
ax1.plot([i.step for i in val_acc], [i.value for i in val_acc], label='val_acc')
ax1.set_xlim(0)
acc = ea.scalars.Items('acc')
ax1.plot([i.step for i in acc], [i.value for i in acc], label='acc')
ax1.set_xlabel("step")
ax1.set_ylabel("")

plt.legend(loc='lower right')
plt.show()
