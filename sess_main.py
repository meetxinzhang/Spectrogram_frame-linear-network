import tensorflow as tf
import numpy as np
import slice_input_producer as ild
from model import The3dcnn_lstm_Model

sess = tf.InteractiveSession()

input_data = ild.InputLocalData(train_file_dir='sounds/', num_epochs=None)
# 训练和测试用的批量本地数据
img_batch, lab_batch = input_data.get_batches(batch_size=50, capacity=7)
print('1111111111111111111111', img_batch.get_shape().as_list())

model = The3dcnn_lstm_Model(160, 50, 8)
logits = model.call(inputs=img_batch, training=True)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lab_batch)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(lab_batch, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    for step in np.arange(2000):
        print("training step: %d" % step)
        print('22222222222222222', sess.run(img_batch))
        if coord.should_stop():
            break
        sess.run(train_step)
        print("accuracy: {}\n".format(accuracy.eval()))
    # Save the variables to disk.
    # args.save()
except tf.errors.OutOfRangeError:
    print("Done!!!")
finally:
    coord.request_stop()
coord.join(threads)

# for i in range(100):
#     batch = get_data_MRI('train_data/', 2, 8)
#     print(batch.shape())
