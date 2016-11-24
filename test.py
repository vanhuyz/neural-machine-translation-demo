from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import zipfile
import random
import collections
import pickle
from six.moves.urllib.request import urlretrieve


# Define language
def grammar(length):
  mygrammar = [1, 0, 2]
  if length <= 0:
    raise ValueError('Length should be >= 1') 
  if length == 1:
    return [0]
  if length == 2:
    return [1,0]
  for i in range(3,length):
    if length % 3 == 1 and i == length - 1:
      next_pos = length - 1
    else:
      next_pos = mygrammar[i-3] + 3
    mygrammar.append(next_pos)
  return mygrammar

def encode(text):
  """ Numberize a sequence """
  words = text.split()
  new_text = ''
  for i in grammar(len(words)):
    new_text += str(len(words[i]))
  return new_text

with open('dicts/dictionary.pickle', 'rb') as handle:
  dictionary = pickle.load(handle)
with open('dicts/reverse_dictionary.pickle', 'rb') as handle:
  reverse_dictionary = pickle.load(handle)

def word2id(word):
  return dictionary.get(word, 0)

MAX_INPUT_SEQUENCE_LENGTH = 10
MAX_OUTPUT_SEQUENCE_LENGTH = 20
PAD_ID = 10
GO_ID = 11
EOS_ID = 12
vocabulary_size = 50000

def construct_graph():
  encoder_inputs = list()
  decoder_inputs = list()
  labels = list()
  weights = list() 

  for _ in range(MAX_INPUT_SEQUENCE_LENGTH):
    encoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
  for _ in range(MAX_OUTPUT_SEQUENCE_LENGTH+1):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=(None,)))
    labels.append(tf.placeholder(tf.int32, shape=(None,)))
    weights.append(tf.placeholder(tf.float32, shape=(None,)))

  feed_previous = True
  learning_rate = tf.placeholder(tf.float32)
  
    # Use LSTM cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  with tf.variable_scope("seq2seq"):
    outputs, states, attentions = tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                          decoder_inputs,
                                                          cell,
                                                          vocabulary_size, # num_encoder_symbols
                                                          13, # num_decoder_symbols
                                                          128, # embedding_size
                                                          feed_previous=feed_previous # False during training, True during testing
                                                        )

  #query = encoder_inputs
  #attention_states = array_ops.concat(1, states)
  #print(attention_states.get_shape())
  #attention = get_attention(query,attention_states)
  loss = tf.nn.seq2seq.sequence_loss(outputs, labels, weights) 
  predictions = tf.pack([tf.nn.softmax(output) for output in outputs])

  tf.scalar_summary('learning rate', learning_rate)
  tf.scalar_summary('loss', loss)
  merged = tf.merge_all_summaries()

  return encoder_inputs, decoder_inputs, labels, weights, learning_rate, feed_previous, outputs, states, attentions, loss, predictions, merged

lstm_size = 256
graph = tf.Graph()
with graph.as_default():
  encoder_inputs, decoder_inputs, labels, weights, learning_rate, feed_previous, outputs, states, attentions, loss, predictions, merged = construct_graph()
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  saver = tf.train.Saver()

class BatchGenerator(object):
  def __init__(self, text, batch_size, global_id = 0):
    self._words = text.split()
    self._text_size = len(text)
    self._batch_size = batch_size
    self._global_id = global_id
  
  def next(self):
    input_sequences = list()
    encoder_inputs = list()
    decoder_inputs = list()
    labels = list()
    weights = list()

    for i in range(self._batch_size):
      #length = random.randint(1,MAX_INPUT_SEQUENCE_LENGTH)
      length = MAX_INPUT_SEQUENCE_LENGTH
      input_words = self._words[self._global_id:self._global_id+length]
      input_word_ids = [word2id(word) for word in input_words]
      
      # reverse list and add padding
      reverse_input_word_ids = [0]*(MAX_INPUT_SEQUENCE_LENGTH-len(input_word_ids)) + input_word_ids[::-1]
      input_sequence = ' '.join(input_words)
      label_sequence = encode(input_sequence)
      label_word_ids = [int(num) for num in label_sequence]
      weight = [1.0]*len(label_word_ids)

      # append to lists
      input_sequences.append(input_sequence)
      encoder_inputs.append(reverse_input_word_ids)
      decoder_inputs.append([GO_ID] + label_word_ids + [PAD_ID]*(MAX_OUTPUT_SEQUENCE_LENGTH-len(label_word_ids)))
      labels.append(label_word_ids + [EOS_ID] + [PAD_ID]*(MAX_OUTPUT_SEQUENCE_LENGTH-len(label_word_ids)))
      weights.append(weight + [1.0] + [0.0]*((MAX_OUTPUT_SEQUENCE_LENGTH-len(weight))))

      # Update global_id
      new_global_id = self._global_id + length
      if new_global_id > len(self._words) - self._batch_size*MAX_INPUT_SEQUENCE_LENGTH:
        self._global_id = 0
      else:
        self._global_id = new_global_id

    return input_sequences, np.array(encoder_inputs).T, np.array(decoder_inputs).T, np.array(labels).T, np.array(weights).T


def id2num(num_id):
  if num_id < 10:
    return str(num_id)
  if num_id == PAD_ID:
    return 'P'
  if num_id == GO_ID:
    return 'G'
  if num_id == EOS_ID:
    return 'E'

def gen_text(predictions):
  text = ''
  for onehot in predictions:
    num = id2num(np.argmax(onehot[0]))
    if num != 'E':
      text += num
    else:
      return text
  return text


with tf.Session(graph=graph) as sess:
  saver.restore(sess, "checkpoints/20161121_model-280000steps.ckpt")
   
  test_feed_dict = dict()
  test_batches = BatchGenerator('neural machine translation', 1)
  current_test_sequences, current_test_encoder_inputs, current_test_decoder_inputs, current_test_labels, current_test_weights = test_batches.next()
  test_feed_dict = {encoder_inputs[i]: current_test_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
  test_feed_dict.update({decoder_inputs[i]: [0.0] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
  test_feed_dict.update({labels[i]: current_test_labels[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
  test_feed_dict.update({weights[i]: [1.0] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
  #test_feed_dict.update({feed_previous: True})
  test_feed_dict.update({learning_rate: 0.1})

  current_test_loss, current_weights, current_test_predictions, test_summary, test_attentions = sess.run([loss, weights, predictions, merged, attentions], feed_dict=test_feed_dict)
      
  print('Test set:')
  print('  Loss       : ', current_test_loss)
  print('  Input            : ', current_test_sequences[0])
  print('  Correct output   : ', ''.join([id2num(n) for n in current_test_labels.T[0]]))
  print('  Generated output : ', gen_text(current_test_predictions))
  print('attns:')
  print(test_attentions)