import numpy as np
import pandas as pd


def get_questions_matrix_sum(questions, nlpProcessor):

	num_samples = len(questions)
	
	word_vec_dim = nlpProcessor(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros((num_samples, word_vec_dim))
	
	for i in xrange(num_samples):
		tokens = nlpProcessor(questions[i])
		for j in xrange(len(tokens)):
			questions_matrix[i,:] += tokens[j].vector

	return questions_matrix



def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
	
	num_samples = len(img_coco_ids)
	feat_dimension = VGGfeatures.shape[0]
	image_matrix = np.zeros((num_samples, feat_dimension))
	for j  in xrange(len(img_coco_ids)):
		image_matrix[j,:] = VGGfeatures[:, img_map[img_coco_ids[j]]]

	return image_matrix



def get_answers_matrix(answers, encoder):
	
	y = encoder.transform(answers)
	num_classes = encoder.classes_.shape[0]
	Y = to_categorical(y, num_classes) 
	return Y


def to_categorical(y, num_classes=None):
	
	y = np.asarray(y, dtype='int32')
	if not num_classes:
		num_classes = np.max(y)+1
	Y = np.zeros((len(y), num_classes))
	for i in range(len(y)):
		Y[i, y[i]] = 1.
	return Y
