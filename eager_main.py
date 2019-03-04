# coding:utf-8
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import input_dataset
import the3dcnn
import model
import rnn
import get_features

tfe.enable_eager_execution()

# sess = tf.InteractiveSession()
#
# input_data = ild.InputLocalData(train_file_dir='sounds/', num_class=8)
#
# # 训练和测试用的批量本地数据
# img_batch, lab_batch = input_data.get_batches(batch_size=3, capacity=6)

#################################

dataset, _ = input_dataset.get_dataset('sounds/', batch_size=50)

# model = the3dcnn.ModelOfDNN()
i = 0


def cal_loss(logits2, lab_batch):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits2)
    loss = tf.reduce_mean(cross_entropy)
    return loss


the_model = model.The3dcnn_lstm_Model(rnn_units=160, batch_size=50, num_class=8)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

for [img_batch, lab_batch] in tfe.Iterator(dataset):
    img_batch = tf.py_func(get_features.get_mp3_tus, [img_batch], tf.float32)
    img_batch = tf.cast(img_batch, tf.float32)

    print(type(img_batch))
    print(type(lab_batch))

    # logits1 = model.output_3dcnn(img_batch)
    # logits2 = rnn.output_lstm(logits1, batch_size=50, hidden_size=160, num_class=8, layer_num=3, keep_prob=1, name='rnn')

    # with tf.GradientTape() as tape:
    #     logits2 = the_model.call(img_batch, True)
    #     loss = cal_loss(logits2, lab_batch)
    #
    # grads = tape.gradient(loss, the_model.variables)
    # optimizer.apply_gradients(zip(grads, the_model.variables))
    logits2 = the_model.call(img_batch, True)
    loss = cal_loss(logits2, lab_batch)
    val_grad_fn = tfe.implicit_value_and_gradients(loss)
    optimizer.apply_gradients(val_grad_fn)

    correct_prediction = tf.equal(tf.argmax(logits2, 1), tf.cast(lab_batch, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    i = i + 1
    print('step: ' + i + 'acc: ' + accuracy)
#################################

# dense_layer = tf.layers.Dense(1)
# def loss(x, y):
#   return tf.reduce_sum(tf.square(dense_layer(x) - y))
#
# # Obtain the gradient function.
# val_grad_fn = tfe.implicit_value_and_gradients(loss)
#
# # Invoke the gradient function with concrete values of x and y.
# x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# y = tf.constant([[10.0], [20.0]])
# value, grads_and_vars = val_grad_fn(x, y)
# print('Value of loss: %s' % value)
#
# # Apply the gradients to Variables.
# optimizer = tf.train.GradientDescentOptimizer(0.1)
# optimizer.apply_gradients(grads_and_vars)

# cnn = the3dcnn.ModelOfDNN()
# logits1 = cnn.output_3dcnn(img_batch)
# logits2 = rnn.output_lstm(logits1, batch_size=3, hidden_size=160, num_class=8, layer_num=3, keep_prob=1, name='rnn')
#
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=lab_batch)
# loss = tf.reduce_mean(cross_entropy)
#
# train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(logits2, 1), tf.cast(lab_batch, tf.int64))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# try:
#     for step in np.arange(100):
#         print("training step: %d" % step)
#         if coord.should_stop():
#             break
#         sess.run(train_step)
#         print("accuracy: {}\n".format(accuracy.eval()))
#
#     # Save the variables to disk.
#     # args.save()
# except tf.errors.OutOfRangeError:
#     print("Done!!!")
# finally:
#     coord.request_stop()
# coord.join(threads)
