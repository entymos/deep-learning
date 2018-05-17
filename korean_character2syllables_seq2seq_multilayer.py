## 한국어 글자 -> 자음모음
## 데이터는 임의의 한글 corpus를 이용해 같은 폴더에 corpus.txt 파일을 생성
## hgtk가 설치되어 있어야함
## tensorflow 1.4

"""
Poech: 1500 Avg. cost =  0.010
Training Done
Testing
넣 -> ㅁㅕE
게 -> ㄱㅔE
종 -> ㄴㅡㄴE
그 -> ㄱㅕE
장 -> ㅇㅏㅇE
손 -> ㄱㅕEE
안 -> ㅇㅏㄴE

Poech: 2000 Avg. cost =  0.008
Training Done
Testing
넣 -> ㅁㅕE
게 -> ㄱㅔE
종 -> ㅇㅗE
그 -> ㄱㅡE
장 -> ㅇㅏㅇE
손 -> ㅁㅓE
안 -> ㅇㅣㅆE
"""

import tensorflow as tf
import numpy as np
import hgtk ## pip install hgtk

## Data Preprocessing ##
corpus = ''
corpus_hangul = ''
letter_syllable_set = []

with open('corpus.txt', 'r', encoding='utf-8-sig') as file:
    texts = file.readlines()
    corpus = ''.join(texts)
    
    for line in corpus[:10000].splitlines():        
        for words in line.split(' '):
            for word in words:
                for letter in word:
                    if hgtk.checker.is_hangul(letter) and len(letter)>0:
                        corpus_hangul += letter
                        letter_syllable_set.append([letter, list(hgtk.letter.decompose(letter))])
            corpus_hangul += ' '
        corpus_hangul += '\n'

letter_syllable_seq_data = []
letter_syllable_vocab = []
for letter_syllable in letter_syllable_set:
	letter_syllable_seq_data.append([letter_syllable[0], ''.join(letter_syllable[1])])
	letter_syllable_vocab.append(letter_syllable[0])
	letter_syllable_vocab.append(''.join(letter_syllable[1]))

char_arr = [c for c in ''.join(set('SEP ' + ''.join(letter_syllable_vocab)))]
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = letter_syllable_seq_data

## Batching ##
def make_batch(seq_data, seq_length):
	input_batch = []
	output_batch = []
	target_batch = []

	for seq in seq_data:
		input = [num_dic[n] for n in seq[0]]
		output = [num_dic[n] for n in ('S' + seq[1]).ljust(seq_length)]
		target = [num_dic[n] for n in (seq[1] + 'E').ljust(seq_length)]

		input_batch.append(input)
		output_batch.append(output)
		
		target_batch.append(target)

	return input_batch, output_batch, target_batch

def next_batch(batch_size, input_data, output_data, target_data):
	import random
	
	I, O, T = input_data[:batch_size], output_data[:batch_size], target_data[:batch_size]
	index_shuf = list(range(len(I)))
	random.shuffle(index_shuf)
	I_shuf, O_shuf, T_shuf = [], [], []
	for idx in index_shuf:
		I_shuf.append(I[idx])
		O_shuf.append(O[idx])
		T_shuf.append(T[idx])

	return I_shuf, O_shuf, T_shuf

## Model ##
learning_rate = 0.001 ## usually 0.0001, 0.001
n_hidden = 256
total_epoch = 100 #2000
n_class = n_input = dic_len
num_layers = 10
dim_embedding = 15
sequence_length = 4
batch_size = 256
dropout_factor = 0.5

## [batch size, time steps, input size]
enc_input = tf.placeholder(tf.int32, [None, None])
dec_input = tf.placeholder(tf.int32, [None, None])
## [batch size, time steps], time steps:글자수:encoder-docker length
targets = tf.placeholder(tf.int32, [None, None])
## for variable length
seq_len = tf.placeholder(tf.int32)
dropout_prob = tf.placeholder(tf.float32)

def get_max_time(tensor):
	time_axis = 0 if False else 1 ## time_axis = 0 if self.time_major else 1
	return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

def build_cell(n_hidden):
	#rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
	rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
	#rnn_cell = tf.contrib.rnn.AttentionCellWrapper(cell=rnn_cell, attn_length=1) ## attn_length:size of attention window
	rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=dropout_prob)
	return rnn_cell

def build_multiple_layers_1(_n_hidden, _num_layers):
	return [build_cell(_n_hidden) for _ in range(_num_layers)]

def build_multiple_layers_2(_hiddens):
	return [build_cell(_size) for _size in _hiddens]

## attention (two way to implement: tf.contrib.rnn.AttentionCellWrapper, tf.contrib.seq2seq.LuongAttention

with tf.variable_scope('encode'):
	enc_embeddings = tf.get_variable(name="enc_embeddings", dtype="float32", shape=[dic_len, dim_embedding]) ## dic_len=vocab_size
	enc_input_emb = tf.nn.embedding_lookup(enc_embeddings, enc_input, name="enc_input_emb")

	enc_cells = tf.nn.rnn_cell.MultiRNNCell(build_multiple_layers_1(n_hidden, num_layers)) ##, state_is_tuple=True)
	#enc_cells = tf.nn.rnn_cell.MultiRNNCell(build_multiple_layers_2([128,256])) ##, state_is_tuple=True)
	## If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...]
	outputs, enc_states = tf.nn.dynamic_rnn(enc_cells, enc_input_emb, dtype=tf.float32, sequence_length=seq_len)

with tf.variable_scope('decode'):
	dec_embeddings = tf.get_variable(name="dec_embeddings", dtype="float32", shape=[dic_len, dim_embedding])
	dec_input_emb = tf.nn.embedding_lookup(dec_embeddings, dec_input, name="dec_input_emb")

	dec_cells = tf.nn.rnn_cell.MultiRNNCell(build_multiple_layers_1(n_hidden, num_layers))
	#dec_cells = tf.nn.rnn_cell.MultiRNNCell(build_multiple_layers_2([128,256])) ##, state_is_tuple=True)
	## If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...]
	outputs, dec_states = tf.nn.dynamic_rnn(dec_cells, dec_input_emb, initial_state=enc_states, dtype=tf.float32, sequence_length=seq_len) ## initial_state: enc to dec

model = tf.layers.dense(outputs, n_class, activation=None) ## model:target_output
prediction = tf.argmax(model, 2)

# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
target_weights = tf.sequence_mask(seq_len, get_max_time(model), dtype=tf.float32)
crossent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
cost = (tf.reduce_sum(crossent * target_weights) / batch_size) ## this step down the errors # by tensorflow MNT

## calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(cost, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0) ## clip_norm usally use 5 or 1
optimizer = tf.train.AdamOptimizer(learning_rate) ## .minimize(cost) -> apply_gradients / sometimes, SGD is better
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

with tf.device('/gpu:0'):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	input_batch, output_batch, target_batch = make_batch(seq_data, sequence_length)

	for epoch in range(total_epoch):
		
		input_batch, output_batch, target_batch = next_batch(batch_size, input_batch, output_batch, target_batch)
		_, loss = sess.run([update_step, cost], feed_dict={enc_input:input_batch, dec_input:output_batch, targets:target_batch, seq_len:sequence_length, dropout_prob:dropout_factor})
		
		print('Poech:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(loss))

	print('Training Done')

def translate(word):
	input_batch, output_batch, target_batch = make_batch([[word, 'S']], sequence_length) ## decoder receives a starting symbol "\<s>"
	result = sess.run(prediction, feed_dict={enc_input:input_batch, dec_input:output_batch, seq_len:sequence_length, dropout_prob:1.0})
	decoded = [char_arr[i] for i in result[0]] ## choose most likely word (greedy)
	translated = ''.join(decoded[:-1]).strip()
	return translated

with tf.device('/gpu:0'):
	print('Testing')

	print('넣 ->', translate('넣'))
	print('게 ->', translate('게'))
	print('종 ->', translate('종'))
	print('그 ->', translate('그'))
	print('장 ->', translate('장'))
	print('손 ->', translate('손'))
	print('안 ->', translate('안'))
