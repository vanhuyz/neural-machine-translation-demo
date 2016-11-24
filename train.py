import os
import tensorflow as tf
import numpy as np
import zipfile
import random
import collections
import pickle
from six.moves.urllib.request import urlretrieve
import datetime


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



# Data
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))


test_size = 1004
test_text = text[:test_size]
train_text = text[test_size:]
train_size = len(train_text)

# Dictionary
vocabulary_size = 50000
def build_dictionary(words):
  count = collections.Counter(words).most_common(vocabulary_size - 2)
  dictionary = dict()
  dictionary['<PAD>'] = 0 
  dictionary['<UNK>'] = 1
  for word, _ in count:
    if word != '<UNK>':
      dictionary[word] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reverse_dictionary

if os.path.exists('dicts/dictionary.pickle'):
  with open('dicts/dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
  with open('dicts/reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
else:
  words = train_text.split()
  dictionary, reverse_dictionary = build_dictionary(words)
  with open('dicts/dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle)
  with open('dicts/reverse_dictionary.pickle', 'wb') as handle:
    pickle.dump(reverse_dictionary, handle)

# BatchGenerator
MAX_INPUT_SEQUENCE_LENGTH = 10
MAX_OUTPUT_SEQUENCE_LENGTH = 20
PAD_ID = 10
GO_ID = 11
EOS_ID = 12

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
      length = random.randint(1,MAX_INPUT_SEQUENCE_LENGTH)
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

batch_size = 16
train_batches = BatchGenerator(train_text, batch_size)
test_batches = BatchGenerator(test_text, 1)

# Utils
def id2num(num_id):
  if num_id < 10:
    return str(num_id)
  if num_id == PAD_ID:
    return 'P'
  if num_id == GO_ID:
    return 'G'
  if num_id == EOS_ID:
    return 'E'

def sampling(predictions):
  return ''.join([id2num(np.argmax(onehot[0])) for onehot in predictions])

def word2id(word):
    return dictionary.get(word, 0)

# Model
lstm_size = 256

def construct_graph(use_attention=True):
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

  feed_previous = tf.placeholder(tf.bool)
  learning_rate = tf.placeholder(tf.float32)

    # Use LSTM cell
  cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  with tf.variable_scope("seq2seq"):
    if use_attention:
      outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                          decoder_inputs,
                                                          cell,
                                                          vocabulary_size, # num_encoder_symbols
                                                          13, # num_decoder_symbols
                                                          128, # embedding_size
                                                          feed_previous=feed_previous # False during training, True during testing
                                                          )
    else: 
      outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                          decoder_inputs,
                                                          cell,
                                                          vocabulary_size, # num_encoder_symbols
                                                          13, # num_decoder_symbols
                                                          128, # embedding_size
                                                          feed_previous=feed_previous # False during training, True during testing
                                                         )
  loss = tf.nn.seq2seq.sequence_loss(outputs, labels, weights) 
  predictions = tf.pack([tf.nn.softmax(output) for output in outputs])

  tf.scalar_summary('learning rate', learning_rate)
  tf.scalar_summary('loss', loss)
  merged = tf.merge_all_summaries()

  return encoder_inputs, decoder_inputs, labels, weights, learning_rate, feed_previous, outputs, states, loss, predictions, merged

graph = tf.Graph()
with graph.as_default():
  encoder_inputs, decoder_inputs, labels, weights, learning_rate, feed_previous, outputs, states, loss, predictions, merged = construct_graph()
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  saver = tf.train.Saver()

# Run session
today_dt = datetime.date.today()
today = today_dt.strftime("%Y%m%d")

with tf.Session(graph=graph) as sess:
  sess.run(tf.initialize_all_variables())
  train_writer = tf.train.SummaryWriter('tensorboard/train', graph)
  test_writer = tf.train.SummaryWriter('tensorboard/test', graph)
  current_learning_rate = 0.1

  for step in range(500001):
    feed_dict = dict()
    current_train_sequences, current_train_encoder_inputs, current_train_decoder_inputs, current_train_labels, current_weights = train_batches.next()
    feed_dict = {encoder_inputs[i]: current_train_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
    feed_dict.update({decoder_inputs[i]: current_train_decoder_inputs[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
    feed_dict.update({labels[i]: current_train_labels[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
    feed_dict.update({weights[i]: current_weights[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
    feed_dict.update({feed_previous: False})

    if step != 0 and step % 50000 == 0:
      current_learning_rate /= 2
    feed_dict.update({learning_rate: current_learning_rate})

    _, current_train_loss, current_train_predictions, train_summary = sess.run([optimizer, loss, predictions, merged], feed_dict=feed_dict)

    train_writer.add_summary(train_summary, step)
    train_writer.flush()
    
    if step % 1000 == 0:
      print('Step %d:' % step)
      print('Training set:')
      print('  Loss       : ', current_train_loss)
      print('  Input            : ', current_train_sequences[0])
      print('  Correct output   : ', ''.join([id2num(n) for n in current_train_labels.T[0]]))
      print('  Generated output : ', sampling(current_train_predictions))
      
      test_feed_dict = dict() 
      current_test_sequences, current_test_encoder_inputs, current_test_decoder_inputs, current_test_labels, current_test_weights = test_batches.next()
      test_feed_dict = {encoder_inputs[i]: current_test_encoder_inputs[i] for i in range(MAX_INPUT_SEQUENCE_LENGTH)}
      test_feed_dict.update({decoder_inputs[i]: current_test_decoder_inputs[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
      test_feed_dict.update({labels[i]: current_test_labels[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})
      test_feed_dict.update({weights[i]: current_test_weights[i] for i in range(MAX_OUTPUT_SEQUENCE_LENGTH+1)})

      test_feed_dict.update({feed_previous: True})
      test_feed_dict.update({learning_rate: current_learning_rate})
      current_test_loss, current_test_predictions, test_summary = sess.run([loss, predictions, merged], feed_dict=test_feed_dict)
      
      print('Test set:')
      print('  Loss       : ', current_test_loss)
      print('  Input            : ', current_test_sequences[0])
      print('  Correct output   : ', ''.join([id2num(n) for n in current_test_labels.T[0]]))
      print('  Generated output : ', sampling(current_test_predictions))
      print('='*50)
      test_writer.add_summary(test_summary, step)
      test_writer.flush()
        
    if step % 10000 == 0:
        # Save the variables to disk.
        save_path = saver.save(sess, "checkpoints/{}_model-{}steps.ckpt".format(today, step))
        print("Model saved in file: %s" % save_path)
