import data
import numpy as np
import scipy
clsNum = 5

def exp(x):
	return np.exp(x)



def compute_posterior(sent, cls, feat2id, weights, sent2feat):
	feature_sets = sent2feat[sent]
	# e^( w_1*f_1 + w_2*f_2 ...)
	nominator = sum( [ weights[(feature, cls)] for feature in feature_sets] )
	# e^(cls1 feature sets) + e^(cls2 feature sets) ...
	denominator = [ sum([weights.get( (feature, label), 0) for feature in feature_sets]) for label in range(clsNum) ]
	posterior = exp(nominator - scipy.misc.logsumexp(denominator))
	return posterior
	
def observed_features(batch_train, batch_cls, sent2feat):
	count = 0	
	for sent in batch_train:
		
def compute_gradient(batch_train, batch_cls, weights, sent2feat):
	observed_featuers = compute_observed_counts(batch_train, batch_cls, sent2feat)
	expected_features = compute_estimated_counts(batch_train, batch_cls, sent2feat)
	return observed_features - expected_features
	
def generate_features(corpus, corpus_cls):
	feat2id = {}
	initial_weights = {}
	feat_id = 0
	sent2feat = {}
	for sent, cls in zip(corpus, corpus_cls):
		prevToken = ''
		feature_set = []
		for tokenIndx, token in enumerate(sent.split()):
			#init_distribution = 1./len(sent.split())
			# unigram
			if token not in feat2id:
				feat2id[(token)] = feat_id
				feat_id +=1
				feature_set.append(token)
			initial_weights[(token,cls)] = 0
			# bigram
			bigram_feat = prevToken + " " + token
			if tokenIndx > 0 and bigram_feat not in feat2id:
				feat2id[(bigram_feat)] = feat_id
				feat_id += 1
				feature_set.append(bigram_feat)
			initial_weights[(bigram_feat,cls)] = 0
			prevToken = token
		sent2feat[sent] = feature_set
	featNum = len(feat2id)
	print("Total number of features throughout all corpus : ", featNum)
	print("Total number of parameters throughout all corpus : ", len(initial_weights))
	return feat2id, initial_weights, sent2feat

def minibatch_training(batch_train, batch_cls, weights, feat2id, sent2feat):
	### calculating errors
	#compute_posterior
	return 0
	#compute_gradient(batch_train, batch_cls, weights, sent2feat)

def training(corpus, corpus_cls, feat2id, weights, sent2feat):
	batch_size = 150
	max_iteration = 1
	weights = np.zeros([clsNum,len(feat2id)])
	#print("weights", weights, " feat2id", len(feat2id))
	train_len = len(corpus)
	total_steps = train_len//batch_size
	
	## shuffling
	def shuffling(corpus, corpus_cls):
		import random
		pair = list(zip(corpus, corpus_cls))
		random.shuffle(pair)
		corpus, corpus_cls = zip(*pair)
		corpus = list(corpus)
		corpus_cls = list(corpus_cls)
		return corpus, corpus_cls
	for i in range(max_iteration):
		# shuffle every epoch
		corpus, corpus_cls = shuffling(corpus, corpus_cls)

		for step in range(total_steps):
			start = step*batch_size
			end = (step+1)*batch_size

			batch_train, batch_cls = corpus[start:end], corpus_cls[start:end]
			#minibatch_training(batch_train, batch_cls, weights, feat2id, sent2feat)
		
	

def main():
	train, train_cls, test, test_cls = data.getData()
	feat2id, weights, sent2feat = generate_features(train, train_cls)
	training(train, train_cls, feat2id, weights, sent2feat)
	
if __name__ == "__main__":
	main()
