#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-7-31"

'''
test tensorflow rnn api
'''

import tensorflow as tf 
import pdb

cell1 = tf.nn.rnn_cell.BasicRNNCell(50)
# cell2 = tf.nn.rnn_cell.BasicLSTMCell(50)
x = tf.random_uniform((2,8,30))#mock data:(batch_size,length,embedding_size)
x1 = tf.unstack(x,axis=1)
print((tf.nn.static_rnn(cell1,x1,dtype=tf.float32)))
'''
(outputs,states)
([<tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_0:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_1:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_2:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_3:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_4:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_5:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_6:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_7:0' shape=(2, 50) dtype=float32>],
 <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_7:0' shape=(2, 50) dtype=float32>)
'''
print(tf.nn.static_bidirectional_rnn(cell1,cell1,x1,dtype=tf.float32))
'''
(outputs,states)
([<tf.Tensor 'concat_0:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_1:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_2:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_3:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_4:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_5:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_6:0' shape=(2, 100) dtype=float32>,
  <tf.Tensor 'concat_7:0' shape=(2, 100) dtype=float32>],
 <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_15:0' shape=(2, 50) dtype=float32>,
 <tf.Tensor 'rnn/rnn/basic_rnn_cell/Tanh_23:0' shape=(2, 50) dtype=float32>)
'''
pdb.set_trace()
print(tf.nn.dynamic_rnn(cell1,x,dtype=tf.float32))
'''
(outputs,states)
(<tf.Tensor 'rnn_1/transpose:0' shape=(2, 8, 50) dtype=float32>,
 <tf.Tensor 'rnn_1/while/Exit_2:0' shape=(2, 50) dtype=float32>)
 '''
print(tf.nn.bidirectional_dynamic_rnn(cell1,cell1,x,dtype=tf.float32))
'''
(outputs,states)
((<tf.Tensor 'bidirectional_rnn_1/fw/fw/transpose:0' shape=(2, 8, 50) dtype=float32>,
  <tf.Tensor 'ReverseV2:0' shape=(2, 8, 50) dtype=float32>),
 (<tf.Tensor 'bidirectional_rnn_1/fw/fw/while/Exit_2:0' shape=(2, 50) dtype=float32>,
  <tf.Tensor 'bidirectional_rnn_1/bw/bw/while/Exit_2:0' shape=(2, 50) dtype=float32>))
'''
