#-*-coding:utf8-*-

__author="buyizhiyou"
__date = "2018-7-30"



import os
import pdb
import re
from collections import Counter
import matplotlib.pyplot as plt

import tensorflow as tf 
from seq2seq import embedding_tied_rnn_seq2seq

from utils import * 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"#choose GPU 1

input_batches = [
    ['Hi What is your name?', 'Nice to meet you!'],
    ['Which programming language do you use?', 'See you later.'],
    ['Where do you live?', 'What is your major?'],
    ['What do you want to drink?', 'What is your favorite beer?']]

target_batches = [
    ['Hi this is Jaemin.', 'Nice to meet you too!'],
    ['I like Python.', 'Bye Bye.'],
    ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
    ['Beer please!', 'Leffe brown!']]


all_input_sentences = []
for input_batch in input_batches:
    all_input_sentences.extend(input_batch)
all_target_sentences = []
for target_batch in target_batches:
    all_target_sentences.extend(target_batch)

enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(all_input_sentences)#enc_vocab:word2idx,enc_reverse_vacab:idx2word,enc_vocab_size:26
dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(all_target_sentences, is_target=True)##dec_vocab:word2idx,dec_reverse_vacab:idx2word,dec_vocab_size:28

#hyperParameters
n_epoch = 2000
hidden_size = 50
enc_emb_size = 20
dec_emb_size = 21
enc_sentence_length=10
dec_sentence_length=11


enc_inputs = tf.placeholder(tf.int32,shape=[None,enc_sentence_length],name='input_sentences')
sequence_lengths = tf.placeholder(tf.int32,shape=[None],name='sentences_length')
dec_inputs = tf.placeholder(tf.int32,shape=[None,dec_sentence_length+1],name='output_sentences')

enc_inputs_t = tf.transpose(enc_inputs,perm=[1,0])
dec_inputs_t = tf.transpose(dec_inputs,perm=[1,0])
labels = tf.one_hot(dec_inputs_t, dec_vocab_size+2)
# labels & logits: [dec_sentence_length+1 x batch_size x dec_vocab_size+2]

# enc_emb_inputs:list(enc_sent_len) of tensor[batch_size x embedding_size]
# Because `static_rnn` takes list inputs
enc_inputs_t = tf.unstack(enc_inputs_t)
dec_inputs_t = tf.unstack(dec_inputs_t)

cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
dec_outputs,state = embedding_tied_rnn_seq2seq(
                            encoder_inputs=enc_inputs_t,
                            decoder_inputs=dec_inputs_t,
                            cell=cell,
                            num_symbols=dec_vocab_size+2,
                            embedding_size=enc_emb_size,
                            output_projection=None,
                            feed_previous=True
)
logits = tf.stack(dec_outputs)
predictions = tf.argmax(logits,axis=2)
predictions = tf.transpose(predictions,[1,0])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits))
# training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_history = []
    for epoch in range(n_epoch):
        all_preds = []
        epoch_loss = 0
        for input_batch,target_batch in zip(input_batches,target_batches):
            input_token_indices = []
            target_token_indices = []
            sentence_lengths = []

            for input_sent in input_batch:
                input_sent,sent_len = sent2idx(input_sent,vocab=enc_vocab,max_sentence_length=enc_sentence_length)
                input_token_indices.append(input_sent)
                sentence_lengths.append(sent_len)
            
            for target_sent in target_batch:
                target_token_indices.append(sent2idx(target_sent,vocab=dec_vocab,max_sentence_length=dec_sentence_length,is_target=True))
    
            batch_preds,batch_loss, _ = sess.run(
                [predictions,loss, training_op],
                feed_dict={
                    enc_inputs: input_token_indices,
                    sequence_lengths: sentence_lengths,
                    dec_inputs: target_token_indices
                })
            loss_history.append(batch_loss)
            epoch_loss += batch_loss
            all_preds.append(batch_preds)
            
        # Logging every 400 epochs
        if epoch % 400 == 0:
            print('Epoch', epoch)
            for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    print('\t', input_sent)
                    print('\t => ', idx2sent(pred, reverse_vocab=dec_reverse_vocab))
                    print('\tCorrent answer:', target_sent)
            print('\tepoch loss: {:.2f}\n'.format(epoch_loss))

show_loss(loss_history)