import data
import math
import numpy as np
import scipy.special
clsNum = 3

def exp(x):
	return np.exp(x)

def compute_conditional_probability(sent, cls, feat2id, weights, sent2feat):
	feature_sets = sent2feat[sent]
	# e^( w_1*f_1 + w_2*f_2 ...)
	nominator = sum( [ weights[cls, feat2id[feature]] for feature in sent2feat[sent]] )
	# e^(cls1 feature sets) + e^(cls2 feature sets) ...
	denominator = [ sum([weights[label, feat2id[feature]] for feature in sent2feat[sent]]) for label in range(clsNum) ]
	posterior = exp(nominator - scipy.special.logsumexp(denominator))

	return posterior

def inference(test, test_cls, feat2id, weights, sent2feat):
	TP = 0.0
	for sent, gold_label in zip(test,test_cls):
		gold_label = gold_label - 1
		predicted = np.zeros([clsNum])
		for test_label in range(clsNum):
			posterior = compute_conditional_probability(sent, test_label, feat2id, weights, sent2feat)
			predicted[test_label] = posterior
		if np.argmax(predicted) == gold_label:
			TP += 1.0
	accuracy = TP/len(test)
	return accuracy		
	
def compute_observed_features(batch_train, batch_cls, sent2feat, feat2id):
	## bag of features under the mini batch
	feature_sets = []
	empirical_count = np.zeros([clsNum, len(feat2id)])
	for sent, cls in zip(batch_train, batch_cls):
		cls = cls-1
		for feat in sent2feat[sent]:
			empirical_count[cls,feat2id[feat]] += 1.0
	return empirical_count	

	
def compute_expected_features(batch_train, batch_cls, sent2feat, feat2id, weights, feat2weights):
	## Similary, bag of feature under the mini batch
	feature_sets = []
	expected_count = np.zeros([clsNum, len(feat2id)])
	neg_likelihood = []
	for sent, cls in zip(batch_train, batch_cls):
		feature_sets += sent2feat[sent]
		cls = cls-1
		conditional_probability = compute_conditional_probability(sent, cls, feat2id, weights, sent2feat)
		for feat in sent2feat[sent]:
			expected_count[cls, feat2id[feat]] += conditional_probability
		if conditional_probability != 0:
			neg_likelihood.append(math.log(conditional_probability))
	return expected_count, sum(neg_likelihood)


def compute_gradient(batch_train, batch_cls, weights, sent2feat, feat2id, feat2weights):
	observed_features = compute_observed_features(batch_train, batch_cls, sent2feat, feat2id)
	expected_features, neg_likelihood = compute_expected_features(batch_train, batch_cls, sent2feat, feat2id, weights, feat2weights)
	#for i, j in zip(observed_features, expected_features):
	#	print(sum(i), sum(j))
	return observed_features - expected_features, neg_likelihood
	
def generate_features(corpus, corpus_cls, feat2id, sent2feat):
	feat2id = feat2id
	feat_id = 0
	sent2feat = sent2feat
	weights_dic = {}
	feat_debug = {}
	for sent, cls in zip(corpus, corpus_cls):
		prevToken = ''
		feature_set = []
		for tokenIndx, token in enumerate(sent.split()):
			#init_distribution = 1./len(sent.split())
			# unigram
			feat_debug[(token, cls)] = 0
			if token not in feat2id:
				feat2id[token] = feat_id
				feat_id +=1
			feature_set.append(token)
			if (token, cls-1) not in weights_dic:
				weights_dic[(token, cls)] = 1
			else:
				# debugging to find duplicate features over the class
				one = weights_dic.get((token, 1), 0)
				two = weights_dic.get((token, 2), 0)
				three = weights_dic.get((token, 3), 0)
				four = weights_dic.get((token, 4), 0)
				five = weights_dic.get((token, 5), 0)
				if one + two + three + four + five>2:
					print(token," ", one, " ", two, " ", three, " ", four , " ", five)
			# bigram
			bigram_feat = prevToken + " " + token
			if tokenIndx > 0 and bigram_feat not in feat2id:
				feat2id[bigram_feat] = feat_id
				feat_id += 1
				feat_debug[(bigram_feat, cls)] = 0
			if tokenIndx > 0:
				weights_dic[(bigram_feat, cls)] = 1
				feature_set.append(bigram_feat)
			prevToken = token
		sent2feat[sent] = feature_set
		
	featNum = len(feat2id)
	return feat2id, sent2feat, weights_dic

	
def training(corpus, corpus_cls, test, test_cls, feat2id, sent2feat, feat2weights):
	batch_size = 5000
	max_iteration = 1000
	weights = np.zeros([clsNum,len(feat2id)])
	#print("weights", weights, " feat2id", len(feat2id))
	train_len = len(corpus)
	total_steps = train_len//batch_size
	print("batch steps : ", total_steps)
	learning_rate = 0.0001
	## shuffling
	def shuffling(corpus, corpus_cls):
		import random
		pair = list(zip(corpus, corpus_cls))
		random.shuffle(pair)
		corpus, corpus_cls = zip(*pair)
		corpus = list(corpus)
		corpus_cls = list(corpus_cls)
		return corpus, corpus_cls
	gradientSum = 0.0
	for i in range(max_iteration):
		# shuffle every epoch
		corpus, corpus_cls = shuffling(corpus, corpus_cls)

		for step in range(total_steps):
			start = step*batch_size
			end = (step+1)*batch_size

			batch_train, batch_cls = corpus[start:end], corpus_cls[start:end]
			new_gradient, neg_loglikelihood = compute_gradient(batch_train, batch_cls, weights, sent2feat, feat2id, feat2weights)
			weights += new_gradient*learning_rate
			gradientSum = sum(sum(new_gradient))

		acc = inference(test, test_cls, feat2id, weights, sent2feat)
		print("Iteration :", i, " Gradient :", sum(sum(new_gradient)), " Neg likelihood", neg_loglikelihood, " ACC :", acc)

def shirink_class(classes):
	new_classes = []
	for cls in classes:
		if cls == 1 or cls == 2:
			new_classes.append(1)
		elif cls == 3 or cls == 4:
			new_classes.append(2)
		else:
			new_classes.append(3)
	return new_classes	

def main():
	train, train_cls, test, test_cls = data.getData()
	train_cls = shirink_class(train_cls)
	test_cls = shirink_class(test_cls)
	feat2id, sent2feat, feat2weights = generate_features(train, train_cls, {}, {})
	feat2id, sent2feat, feat2weights = generate_features(test, test_cls, feat2id, sent2feat)
	print("total feature numbers : ", len(feat2id))
	training(train, train_cls, test, test_cls, feat2id, sent2feat, feat2weights)

if __name__ == "__main__":
	main()
