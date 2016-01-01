import sys
import numpy as np
import argparse 
import scipy.io
import theano
import theano.tensor as T
import lasagne
from sklearn import preprocessing
from spacy.en import English
import time

sys.path.append('/home/mayank/visual_question_ans/tools/')

from tools2 import selectFrequentAnswers, grouper
from feature_extraction import get_questions_matrix_sum, get_images_matrix, get_answers_matrix, to_categorical


def iterate_minibatches(questions, answers, images, batchsize):       
	assert len(questions) == len(answers)
	for start_idx in range(0, len(questions)-batchsize+1, batchsize):
		examples = slice(start_idx, start_idx+batchsize)
		yield questions[examples], answers[examples], images[examples]




def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units', type=int, default=512)
	parser.add_argument('-num_hidden_layers', type=int, default=2)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation', type=str, default='rectified')
	parser.add_argument('-language_only', type=bool, default= False)
	parser.add_argument('-num_epochs', type=int, default=20)
	parser.add_argument('-model_save_interval', type=int, default=10)
	parser.add_argument('-batch_size', type=int, default=128)
	args = parser.parse_args()		


	questions_train = open('../Datasets/VQA/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../Datasets/VQA/preprocessed/answers_train2014_top1000.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../Datasets/VQA/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../Datasets/VQA/coco/vgg_feats.mat'

	questions_val = open('../Datasets/VQA/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
	answers_val = open('../Datasets/VQA/preprocessed/answers2014_top1000.txt', 'r').read().decode('utf8').splitlines()
	images_val = open('../Datasets/VQA/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()


	maxAnswers = 1000
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, maxAnswers)

	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	num_classes = len(list(labelencoder.classes_))

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print "Features loaded from VGG pretrained model"
	img_ids = open('/home/mayank/visual_question_ans/features/coco_vgg_IDMap.txt').read().splitlines()
	id_map = {}
	for id_pair in img_ids:
		id_pair_split = id_pair.split()
		id_map[id_pair_split[0]] = int(id_pair_split[1])

        # The nlp processor
	nlp = English()
	print 'loaded word2vec features...'



	####################Variables###########

	input_var = T.matrix('inputs')
	target_var = T.matrix('targets')

	#######################################

	
	img_dim = 4096
	word_vec_dim = 300

	####################Model############

	nonlin = lasagne.nonlinearities.rectify    ###Change activation here
	depth = args.num_hidden_layers
	num_hidden = args.num_hidden_units
	
	#Input Layer
	network = lasagne.layers.InputLayer(shape=(None, img_dim + word_vec_dim), input_var=input_var)
	network = lasagne.layers.dropout(network, p=0.2)

	#Hidden Layers
	for num in range(depth):
		network = lasagne.layers.DenseLayer(network, num_hidden, nonlinearity=nonlin)
		network = lasagne.layers.dropout(network, p=0.5)

	#Output Layer
	softmax = lasagne.nonlinearities.softmax
	network = lasagne.layers.DenseLayer(network, num_classes, nonlinearity=softmax)

	print 'Model Assembled...'
	###########################################################

	
	###################Prediction and loss####################

	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()

	val_prediction = lasagne.layers.get_output(network, deterministic=True)
	val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var)
	val_loss = val_loss.mean()
	val_acc = T.mean(T.eq(val_prediction, target_var), dtype=theano.config.floatX)


	###################################################

	####################Updates##################################
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.momentum(loss, params, learning_rate=0.01)
	##############################################################

	#####Compilation#######
	print 'compiling model....'
	train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
	print 'model compiled!'


	###################Training######################################
	print 'Training started.........'


	for epoch in range(args.num_epochs):
	
		print('Current Epoch:' + str(epoch))
		##We shuffle the data before training
		index_shuff = range(len(questions_train))
		np.random.shuffle(index_shuff)
		questions_train = [questions_train[i] for i in index_shuff]
		answers_train = [answers_train[i] for i in index_shuff]
		images_train = [images_train[i] for i in index_shuff]

		train_err = 0
		train_batches = 0
		#start_time = time.time()

    	

		for ques_batch, ans_batch, img_batch in iterate_minibatches(questions_train, answers_train, images_train, args.batch_size):
		

			X_ques_batch = get_questions_matrix_sum(ques_batch, nlp)
			X_img_batch = get_images_matrix(img_batch, id_map, VGGfeatures)
			X_batch = np.hstack((X_ques_batch, X_img_batch))

			Y_batch = get_answers_matrix(ans_batch, labelencoder)


			train_err += train_fn(X_batch, Y_batch)
            

			train_batches += 1
			if train_batches%100 == 0:
				print train_batches
				print float(train_err)/float(train_batches)

		
		####Validation accuracy
		val_err = 0
		val_acc = 0
		val_batches = 0
		for ques_val, ans_val, img_val in iterate_minibatches(questions_val, answers_val, images_val, 256)		
			



if __name__ == "__main__":
	main()
			





















