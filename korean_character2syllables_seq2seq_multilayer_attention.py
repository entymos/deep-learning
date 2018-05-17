## 한국어 단어 -> 다음 단어
## 데이터는 임의의 한글 corpus를 이용해 같은 폴더에 corpus.txt 파일을 생성
## hgtk가 설치되어 있어야함
## tensorflow 1.4

"""
Poech: 1500 Avg. cost =  0.022
Training Done
Testing (GreedyEmbeddingHelper)
호텔 -> 앞에서
대사관에 -> 호께
앞에서 -> 러는
급히 -> 뛰어
줄줄이 -> 아는
체크 -> 아웃
"""

## refer to https://www.tensorflow.org/tutorials/seq2seq
## refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/api_guides/python/contrib.seq2seq.md

import tensorflow as tf
import numpy as np
import hgtk #pip install hgtk
from tensorflow.python.layers import core as layers_core
#print(tf.__version__)

## Data Preprocessing ##
corpus = ''
letters_letters_set = []

with open('corpus.txt', 'r', encoding='utf-8-sig') as file:
	texts = file.readlines()
	corpus = ''.join(texts)
	
	for line in corpus[:10000].splitlines():
		sp_line = line.split(' ')
		for words_itr, _ in enumerate(sp_line[:-1]):
			if hgtk.checker.is_hangul(sp_line[words_itr]) and hgtk.checker.is_hangul(sp_line[words_itr+1]):
				letters_letters_set.append([sp_line[words_itr], sp_line[words_itr+1]])        

letters_letters_seq_data = []
letters_letters_vocab = []

for letters_letters in letters_letters_set:
	letters_letters_seq_data.append(letters_letters)
	letters_letters_vocab.append(letters_letters[0])
	letters_letters_vocab.append(letters_letters[1])

char_arr = [c for c in ''.join(set('SEP ' + ''.join(letters_letters_vocab)))]
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = letters_letters_seq_data

## Batching ##
def make_batch(seq_data, enc_len, dec_len):
	input_batch = []
	output_batch = []
	target_batch = []

	for seq in seq_data:
		input = [num_dic[n] for n in seq[0].ljust(enc_len)]
		output = [num_dic[n] for n in ('S' + seq[1]).ljust(dec_len)]
		target = [num_dic[n] for n in (seq[1] + 'E').ljust(dec_len)]

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
batch_size = 256
total_epoch = 1500 #1500 #2000
learning_rate = 0.001 ## usually 0.0001, 0.001
n_hidden = 512 #256 ## =num_units
vocab_size = n_class = n_input = dic_len
num_layers = 10
dim_embedding = 15
encoder_length = 10 ## sequence_length
decoder_length = 10 ## sequence_length
dropout_factor = 0.5
max_gradient_norm = 5.0
attention_layer_depth = 8
use_attention = True
beam_width = 1
GO_SYMBOL = 'S'
END_SYMBOL = 'E'

## [batch size, time steps], time steps:글자수:en/decoder_length, time_major->reverse
enc_input = tf.placeholder(tf.int32, [None, None], name="enc_input")
dec_input = tf.placeholder(tf.int32, [None, None], name="dec_input")
## =target_labels, [batch size, time steps], time steps:글자수:decoder_length
targets = tf.placeholder(tf.int32, [None, None])

## for variable length
dropout_prob = tf.placeholder(tf.float32)
enc_len = tf.placeholder(tf.int32)
dec_len = tf.placeholder(tf.int32)
dec_lens = tf.placeholder(tf.int32, shape=(batch_size), name="decoder_lengths") #shape=(batch_size)

def get_max_time(tensor):
	## time_axis = 0 if self.time_major else 1 # [batch size, time steps]
	time_axis = 0 if False else 1 
	return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

def get_batch_size(tensor):
	return tensor.shape[0].value or tf.shape(tensor)[0]

# def get_num_units(tensor):
# 	## [batch_size, max_time, num_units]
# 	return tensor.shape[2].value or tf.shape(tensor)[2]

def build_cell(n_hidden):
	#rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
	rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
	#rnn_cell = tf.contrib.rnn.AttentionCellWrapper(cell=rnn_cell, attn_length=1) ## attn_length:size of attention window
	rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=dropout_prob)
	return rnn_cell

def build_multiple_layers_1(_n_hidden, _num_layers):
	return [build_cell(_n_hidden) for _ in range(_num_layers)]

with tf.variable_scope('encode'):
	enc_embeddings = tf.get_variable(name="enc_embeddings", dtype="float32", shape=[vocab_size, dim_embedding]) ## dic_len=vocab_size
	enc_input_emb = tf.nn.embedding_lookup(enc_embeddings, enc_input, name="enc_input_emb")

	#enc_cells = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
	enc_cells = tf.nn.rnn_cell.MultiRNNCell(build_multiple_layers_1(n_hidden, num_layers))
	
	## If time_major == False (default), outputs: [batch_size, max_time, cell_state_size], states:[batch_size, cell_state_size]
	enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cells, enc_input_emb, dtype=tf.float32, sequence_length=enc_len)

with tf.variable_scope('decode'):
	dec_embeddings = tf.get_variable(name="dec_embeddings", dtype="float32", shape=[vocab_size, dim_embedding])
	dec_input_emb = tf.nn.embedding_lookup(dec_embeddings, dec_input, name="dec_input_emb")
	#dec_cells = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
	dec_cells = tf.nn.rnn_cell.MultiRNNCell(build_multiple_layers_1(n_hidden, num_layers))
	
	initial_state = [state for state in enc_states]

	if use_attention:
		## If time_major: For the attention mechanism, we need to make sure the "memory" passed in is batch major
		## so, we need to transpose attention_states like 'attention_states = tf.transpose(enc_outputs, [1,0,2])'
		attention_states = enc_outputs ## attention_states should be [batch_size, max_time, num_units]
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=n_hidden, memory=attention_states, memory_sequence_length=None)
		dec_cells = tf.contrib.seq2seq.AttentionWrapper(dec_cells, attention_mechanism, attention_layer_size=attention_layer_depth)
		
		# attn_cell = tf.contrib.rnn.DeviceWrapper(dec_cells, "/device:GPU:0")
		# top_cell = tf.contrib.rnn.DeviceWrapper(tf.nn.rnn_cell.BasicLSTMCell(128), "/device:GPU:0")
		# dec_cells = tf.nn.rnn_cell.MultiRNNCell([attn_cell, top_cell]) #multi_cell

		initial_state = dec_cells.zero_state(get_batch_size(attention_states), tf.float32).clone(cell_state=enc_states)
	else:
		initial_state = enc_states

	decoder_initial_state = initial_state

	## Architecture figure in https://www.tensorflow.org/tutorials/seq2seq
	## TrainingHelper can be substituted with GreedyEmbeddingHelper to do greedy decoding
	projection_layer = layers_core.Dense(vocab_size, use_bias=False)	
	
	helper = tf.contrib.seq2seq.TrainingHelper(dec_input_emb, sequence_length=dec_lens, time_major=False)
	#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,start_tokens=tf.tile([num_dic['S']], [batch_size]), end_token=num_dic['E'])
	
	# Decoder and decode to train
	decoder = tf.contrib.seq2seq.BasicDecoder(dec_cells, helper, decoder_initial_state, output_layer=projection_layer)
	# Dynamic Decoding
	maximum_iterations = tf.round(tf.reduce_max(decoder_length) * decoder_length)
	final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, output_time_major=False, maximum_iterations=maximum_iterations)
	
model = final_outputs.rnn_output ## model:target_output,logits
prediction = tf.argmax(model, 2)

# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
crossent_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
target_weights = tf.sequence_mask(dec_len, get_max_time(model), dtype=tf.float32)
cost = (tf.reduce_sum(crossent_loss * target_weights) / batch_size) ## this step down the errors # by tensorflow MNT

## calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(cost, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=max_gradient_norm) ## clip_norm usally use 5 or 1

## Optimization
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate) ## .minimize(cost) -> apply_gradients / sometimes, SGD is better
train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

with tf.device('/gpu:0'):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	input_batch, output_batch, target_batch = make_batch(seq_data, encoder_length, decoder_length)
	print('Training Start')
	for epoch in range(total_epoch):		
		input_batch, output_batch, target_batch = next_batch(batch_size, input_batch, output_batch, target_batch)
		feed_dict = {
			enc_input:input_batch,
			dec_input:output_batch,
			targets:target_batch,
			dropout_prob:dropout_factor,
			enc_len:encoder_length,
			dec_len:decoder_length,
			dec_lens: np.ones((batch_size), dtype=int) * decoder_length
			}

		_, loss = sess.run([train_op, cost], feed_dict=feed_dict)
		
		print('Poech:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(loss))
	print('Training Done')

def translate(word):
	with tf.device('/cpu:0'):
		# Inference-Decoding using GreedyEmbeddingHelper
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,start_tokens=tf.fill([batch_size], num_dic['S']), end_token=num_dic['E']) # tf.tile([num_dic['S']], [batch_size])
		decoder = tf.contrib.seq2seq.BasicDecoder(dec_cells, helper, initial_state, output_layer=projection_layer)
		maximum_iterations = tf.round(tf.reduce_max(decoder_length) * decoder_length)
		final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, output_time_major=False, maximum_iterations=maximum_iterations)
		model = final_outputs.rnn_output # model = tf.layers.dense(final_outputs, n_class, activation=None) ## model:target_output
		prediction = tf.argmax(model, 2)

		input_batch, output_batch, target_batch = make_batch([[word, 'S'] for _ in range(batch_size)], encoder_length, decoder_length) ## decoder receives a starting symbol "\<s>"
		feed_dict = {
			enc_input:input_batch,
			dec_input:output_batch,
			dropout_prob:1.0,
			enc_len:encoder_length,
			dec_len:decoder_length
		}
		result = sess.run(prediction, feed_dict=feed_dict)
		decoded = [char_arr[i] for i in result[0]] ## choose most likely word (greedy)
		translated = ''.join(decoded[:-1]).strip()
	return translated

def translate_inference(word):
	with tf.device('/cpu:0'):
		# Inference-Decoding using Beam Search / Replicate encoder infos beam_width times
		decoder = tf.contrib.seq2seq.BeamSearchDecoder(
				cell=dec_cells,
				embedding=dec_embeddings,
				start_tokens=tf.fill([batch_size], num_dic['S']),
				end_token=num_dic['E'],
				initial_state=decoder_initial_state,
				beam_width=beam_width, #beam_with>1보다크면 에러남 (tile batch 사용하면 안날거 같은데 tile batch가 자꾸 에러남 --> 버전업하면 나중에 테스트해봐야지)
				output_layer=projection_layer,
				length_penalty_weight=0.0
			)

		maximum_iterations = tf.round(tf.reduce_max(decoder_length) * decoder_length)
		final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False, output_time_major=False, maximum_iterations=maximum_iterations)
		prediction = final_outputs.predicted_ids

		input_batch, output_batch, target_batch = make_batch([[word, 'S'] for _ in range(batch_size)], encoder_length, decoder_length) ## decoder receives a starting symbol "\<s>"
		feed_dict = {
			enc_input:input_batch,
			dropout_prob:1.0,
			enc_len:encoder_length,
			dec_len:decoder_length
		}
		result = sess.run(prediction, feed_dict=feed_dict) #results looks [[][][]], [[][][]], ...
		decoded = [char_arr[j[0]] for j in result[0]]
		translated = ''.join(decoded[:-1]).strip()

		return translated

print('Testing (GreedyEmbeddingHelper)')

print('호텔 ->', translate('호텔')) # 앞에서
print('대사관에 ->', translate('대사관에')) # 연락
print('앞에서 ->', translate('앞에서')) # 만납시다
print('급히 ->', translate('급히')) # 뛰어
print('줄줄이 ->', translate('줄줄이')) # 모가지에요
print('체크 ->', translate('체크')) # 아웃
