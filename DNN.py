# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import data_process as dt

'''
    yxr 4/18/2018
    '''

batch_size = 128
# input*data*weights + biases
f=dt.input_data()
def neural_network_model(data, activation, hiddenLayerNodes):  # 模型主体
    # data输入数据,
    # activition激活函数，
    # hiddenLayerNodes每层节点数，如[3,10,10], 注意第一个数为输入层，宽为3
    a = data
    z = 0
    for i in range(len(hiddenLayerNodes) - 1):  # 前馈传递，通过每个隐含层
        z = tf.add(tf.matmul(a, tf.Variable(tf.random_normal([hiddenLayerNodes[i], hiddenLayerNodes[i + 1]], mean=0))),
                   tf.Variable(tf.random_normal([hiddenLayerNodes[i + 1]])))
        a = activation(z)  # rectified

    softMax_layer = {'weights': tf.Variable(tf.random_normal([hiddenLayerNodes[-1], 2], mean=0)),
                     'biases': tf.Variable(tf.random_normal([2]))}

    output = tf.nn.softmax(tf.matmul(a, softMax_layer['weights']) + softMax_layer['biases'])
    return output


x = tf.placeholder('float', [None, 5000])
y = tf.placeholder('float', [None, 2])
prediction = neural_network_model(x, tf.nn.tanh, [5000, 1000, 1000, 1000, 1000])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
sess = tf.Session()
    
saver = tf.train.Saver()
saver.restore(sess, 'model2/dnn_10000.ckpt-10000')  # 恢复保存的模型


def train_neural_network(continune_train=False):
    if continune_train is False:              # 训练
        global_step = tf.Variable(0)
        global f
        learning_rate = tf.train.exponential_decay(0.0001, global_step, decay_steps=500, decay_rate=0.98,
                                               staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        epochs = 10001

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            for epoch in range(epochs):
                global_step = epoch
                epoch_loss = 0
                for _ in range(int(f.train_num / batch_size) + 1):
                    epoch_x, epoch_y = f.get_train_data()
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                # print epoch_loss
                print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
                if epoch % 2000 == 0:
                    saver.save(sess, 'model2/dnn_' + str(epoch) + '.ckpt', global_step)
                if epoch % 500 == 0:
                    all_acc = 0
                    all_acc=accuracy.eval({x: f.get_test_data()[0], y: f.get_test_data()[1]}, session=sess)
                    print "all accuracy=" + str(all_acc )
    else:
        with tf.Session() as sess:
            global_step = tf.Variable(2000)
            
            learning_rate = tf.train.exponential_decay(0.0001, global_step, decay_steps=1000, decay_rate=0.9,
                                                   staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            epochs = 10001

            saver = tf.train.Saver()
            saver.restore(sess, 'model2/dnn_2000.ckpt-2000')  # 恢复保存的模型,继续训练
            f = dt.input_data()
            for epoch in range(0, epochs):
                global_step = epoch
                epoch_loss = 0
                for _ in range(int(f.train_num / batch_size) + 1):
                    epoch_x, epoch_y = f.get_train_data()
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                    # print epoch_loss
                print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)
                if epoch % 1000 == 0:
                    saver.save(sess, 'model2/dnn_' + str(epoch) + '.ckpt', global_step)
                if epoch % 500 == 0:
                    all_acc = 0
                    all_acc=accuracy.eval({x: f.get_test_data()[0], y: f.get_test_data()[1]}, session=sess)
                    print "all accuracy=" + str(all_acc )


def test_given_data(d):  # 读取保存模型，测试
    result = accuracy.eval({x: d[0], y: d[1]}, session=sess)
    print ("accuracy on given data : ", result)
    return result

def get_benign_prob(d):  # 输出无害几率
    result = prediction.eval({x: d},session=sess)
    return result[0]




def test_all():
    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, 'model2/dnn_2000.ckpt-2000')  # 恢复保存的模型
    all_acc = 0

    all_acc=accuracy.eval({x: f.get_test_data()[0], y: f.get_test_data()[1]}, session=sess)
    print "all accuracy=" + str(all_acc )





