import data
import numpy as np

clsNum = 5

def exp(x):
	return np.exp(x)

def extract_nominators(features, weights, review):
	nominators = {}
	denominator = 0.0
	for cls in range(1, clsNum+1):
		prevtoken = ''
		nominator = 0
		for token in review.split():
			unigram = (token, cls)
			if features.get(unigram, 0) != 0:
				nominator += weights[unigram]
			bigram = (prevtoken + " " + token, cls)
			if features.get(bigram, 0) != 0:
				nominator += weights[bigram]
			prevtoken = token
		nominator = exp(nominator)	
		nominators[cls] = nominator
		denominator += nominator
	return nominators, denominator


def inference(features, weights, review):
	nominators, denominator = extract_nominators(features, review)
	posterior = np.zeros([clsNum])
	for cls, nomi in enumerate(nominators):
		posterior[cls] = nomi/denominator
	predicted = np.argmax(posterior)
			

def generate_features(corpus, corpus_cls):
	feat = {}
	initial_weights = {}

	for sent, cls in zip(corpus, corpus_cls):
		prevToken = ''
		for tokenIndx, token in enumerate(sent.split()):
			init_distribution = 1./len(sent.split())
			# unigram
			feat[(token,cls)] = 1
			initial_weights[(token,cls)] = init_distribution
			# bigram
			bigram_feat = prevToken + " " + token
			if tokenIndx > 0 and bigram_feat not in feat:
				feat[(bigram_feat, cls)] = 1
				initial_weights[(bigram_feat,cls)] = init_distribution
			prevToken = token
	featNum = len(feat)
	print("Total number of features throughout all corpus : ", featNum)

def main():
	train, train_cls, test, test_cls = data.getData()
	generate_features(train, train_cls)
	
if __name__ == "__main__":
	main()
