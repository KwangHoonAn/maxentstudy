import pandas as pd

import re
stopwords = ['i', 'i\'ll', 'i\'ve', 'i\'d', 'i\'m', 't', 's', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','only', 'own', 'same', 'so', 'than', 'too', 'very','now', 'd', 'll', 'm', 'o', 're', 've', 'y']
#review_id                 user_id             business_id  stars        date                                               text  useful  funny  cool
#5    20187
#4    12876
#3     6527
#1     6263
#2     4147
def cleanData(review):
	import string
	punctuations = string.punctuation
	review = re.sub('\n', ' ', review.lower())
	review = re.sub('[,.;@#?!&$-_"]+\ *', " " ,review)
	review = ' '.join( [token for token in review.split() if token not in stopwords] )
	return review	
def getData():
	yelp = pd.read_csv('yelp_review.csv', nrows=50000)
	yelp = yelp.drop('review_id', 1)
	yelp = yelp.drop('user_id', 1)
	yelp = yelp.drop('business_id', 1)
	yelp = yelp.drop('date', 1)
	yelp = yelp.drop('useful', 1)
	yelp = yelp.drop('funny', 1)
	yelp = yelp.drop('cool', 1)
	yelp = yelp[["text", "stars"]]
	print(yelp['stars'].value_counts())
	train = []
	test = []

	for cls in yelp.groupby('stars'):
		train.append(cls[1][:3000])
		test.append(cls[1][3000:4000])
	_train = []
	_train_label = []
	_test = []
	_test_label = []
	for indx in range(5):
		for data in train[indx]['text']:
			data = cleanData(data)
			_train.append(data)
		for label in train[indx]['stars']:
			_train_label.append(label)
		for data in test[indx]['text']:
			data = cleanData(data)
			_test.append(data)
		for label in test[indx]['stars']:
			_test_label.append(label)

	print("Train # :", len(_train))
	print("Test # :", len(_test))
	assert (len(_train) == len(_train_label)), "train len must be same"
	assert (len(_test) == len(_test_label)), "test len must be same"	
	return _train, _train_label, _test, _test_label	

