import numpy as np
import time
import tensorflow as tf
sess = tf.Session()
with sess.as_default():
    model_output = np.random.random((52, 52)).astype(np.float32)
    print(model_output[43, 20])
    action = [[43, 20]]
    action_space = (52, 52)
    action_flattened_index = action[0][0] * action_space[1] + action[0][1]
    print('action flattened index:', action_flattened_index)
    ind = tf.one_hot(action_flattened_index, action_space[0] ** 2).eval()
    flattened_output = tf.reshape(model_output, [-1]).eval()
    print('flattened_output', flattened_output)
    pred_Q = tf.reduce_sum(tf.multiply(flattened_output, ind), axis=0).eval()
    print('pred_Q', pred_Q)

    value = 2
    advantage = [[2, 3, 4, 5],
                 [5, 6, 7, 8]]
    reu_mean = (tf.reduce_mean(advantage)).eval()
    print('reumean', reu_mean)
    output = (value + tf.subtract(advantage, tf.reduce_mean(advantage))).eval()
    print(output)
    tt = tf.Variable(np.zeros(8), dtype=tf.int32)
    sess.run(tf.global_variables_initializer())
    # tf.initialize_variables([tt])

    # print(tt.eval())
    # print(tf.reshape(advantage, shape=(-1, )).shape)
    loss = tf.reduce_sum(tt.assign(tf.reshape(output, [-1])))

    print(sess.run(loss))